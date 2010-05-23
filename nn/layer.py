"""
Created on 11.05.2010

@author: RevEn

Defines base types of layers: Layer, InputLayer, OutputLayer.
Example of usage:

    Importing PyOpenCL and creating context and queue
>>> from opencl import *
>>> ocl = OpenCL( pyopencl.create_some_context( ) )

    Creating input layer with 2 neurons, one hidden layer with 3 neurons
    and output layer with one neuron
>>> i = InputLayer( 2, ocl )
>>> h = Layer( 3, ocl )
>>> o = OutputLayer( 1, ocl )

    Link them together, link only one neuron from hidden layer to output layer, also
    link input layer directly with output. The overall structure will be following:
    
          H
    I - - H -
      \/  H  \ - O
      /\     /
    I - - - -
    
    2 neurons from input layer linked to 3 neurons of hidden layer, and
    one neuron of output layer -> hidden layer has (2+1)*3=9 links
    
    only one neuron (second) of hidden layer is linked to one neuron of
    output layer -> output layer has 1+2+1=4 links
    
    note additional links to each layer are added due to polarization
    
>>> i.link_next( h, 0, 2 )
>>> h.link_next( o, 1, 1 )
>>> i.link_next( o, 0, 2 )
>>> nnc = ExecutionContext( i, o )
>>> nnc.total_neurons
6
>>> nnc.total_weights
19
>>> nnc.total_inputs
7

    Setting up inputs for entire neural network
>>> i.set_inputs( numpy.array( ( 1, 2 ), numpy.float32 ) )

    Setting up weights of links.
>>> i.set_weights( numpy.array( ( 0, 1, 0, 0, 0, 1 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( 0, 1, 2, 3, 4, -5, 6, 7, 8 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( -5, 10, -0.3, -0.2 ), numpy.float32 ) )

    Start simulation
>>> i.process()
>>> i.get_outputs( )
array([ 0.58277857,  0.87005842], dtype=float32)
>>> h.get_outputs( )
array([ 0.91355115,  0.57427275,  1.        ], dtype=float32)
>>> o.get_outputs( )
array([ 0.25671202], dtype=float32)

"""

import numpy
import pyopencl

class ExecutionContext( object ):
    """
    Holds necessary data (buffers, kernels) for entire neural network that are using
    during inputs processing.
    """

    def __init__( self, input_layer, output_layer, allow_training = False ):
        """
        Creates all necessary buffers.
        
        @param input_layer
            Input layer of neural network.
            
        @param output_layer
            Output layer of neural network.
            
        @param allow_training
            if True then some buffers would have read-write access to allow store training data
        """
        self.opencl = input_layer.opencl
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.total_neurons = 0
        self.total_weights = 0
        self.total_inputs = 0       # total inputs to neurons, without polarization link

        ll = [ input_layer ]

        while ll:
            l = ll.pop()

            #process layer
            l.weights_count = l.neuron_count * l.inputs_per_neuron
            l.weights_offset = self.total_weights
            l.neurons_offset = self.total_neurons
            l.inputs_offset = self.total_inputs
            l.context = self
            l.processed = True

            self.total_weights += l.weights_count
            self.total_neurons += l.neuron_count
            self.total_inputs += l.inputs_per_neuron - 1

            for k in l.next_layers:
                if k[0].processed == False:
                    ll.append( k[0] )

        input_layer.reset_processed()

        if allow_training:
            fl = pyopencl.mem_flags.READ_WRITE
        else:
            fl = pyopencl.mem_flags.READ_ONLY

        self.inputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_ONLY, self.total_inputs * 4 )
        self.outputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.WRITE_ONLY, self.total_neurons * 4 )
        self.weights_buf = pyopencl.Buffer( self.opencl.context, fl, self.total_weights * 4 )

        if allow_training:
            self.gradient_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.total_weights * 4 )
            self.errors_backpropagation_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.total_neurons * 4 )

class Layer( object ):
    """
    Defines one layer of neural network.
    """

    def __init__( self, neuron_count, opencl ):
        """
        Constructs layer with specified count of neurons.
        
        @param neuron_count
            Total count of all neurons in the layer.
            
        @param opencl
            OpenCL data.
            
        @see OpenCL
        """

        self.neuron_count = neuron_count
        self.opencl = opencl
        self.prev_layers = []
        self.next_layers = []
        self.processed = False
        self.inputs_per_neuron = 1      # polarization input is always exists

    def link_next( self, next_layer, this_start_neuron, this_neurons_count ):
        """
        Links this layer with next layer.
        
        @param next_layer
            Layer to link this layer with. Part of neurons from this layer
            will be linked with all neurons from next layer, each with each.
        @param this_start_neuron, this_neurons_count
            Range of neurons that will be linked with next layer.
                        
        @see finilize_links
        """

        self.next_layers.append( ( next_layer, this_start_neuron, this_neurons_count ) )
        next_layer.prev_layers.append( ( self, this_start_neuron, this_neurons_count ) )
        next_layer.inputs_per_neuron += this_neurons_count

    def set_weights( self, weights ):
        """
        Set weights for entire layer.
        
        @param weights
            NumPy.NDArray of float32 values, size equals to inputs_per_neuron * neuron_count
        """
        pyopencl.enqueue_write_buffer( 
            self.opencl.queue, self.context.weights_buf, weights,
            device_offset = self.weights_offset * 4, is_blocking = True
            )

    def get_weights( self ):
        """
        Returns weights.
        """
        weights = numpy.ndarray( [ self.weights_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context.weights_buf, weights,
            device_offset = self.weights_offset * 4, is_blocking = True
            )
        return weights

    def get_inputs( self ):
        """
        Returns inputs.
        """
        inputs = numpy.ndarray( [ self.inputs_per_neuron - 1 ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context.inputs_buf, inputs,
            device_offset = self.inputs_offset * 4, is_blocking = True
            )
        return inputs

    def get_outputs( self ):
        """
        Wait for outputs.
        """
        outputs = numpy.ndarray( [ self.neuron_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context.outputs_buf, outputs,
            device_offset = self.neurons_offset * 4, is_blocking = True
            )
        return outputs

    def reset_processed( self ):
        """
        Recursively reset processed flag on all linked layers.
        """
        self.processed = False
        for l in self.next_layers:
            l[0].reset_processed()

    def process( self ):
        """
        Process signal by this layer.
        
        Invokes OpenCL program that produces output array in background.
        """

        # ensure that all previous layers are processed
        for l in self.prev_layers:
            if not l[0].processed:
                return

        i_s = 0
        for l in self.prev_layers:
            pyopencl.enqueue_copy_buffer( 
                self.opencl.queue, l[0].context.outputs_buf, self.context.inputs_buf,
                byte_count = l[2] * 4,
                src_offset = ( l[0].neurons_offset + l[1] ) * 4,
                dst_offset = ( self.inputs_offset + i_s ) * 4
                )
            i_s += l[2]

        #process layer
        self.opencl.kernel_process_layer( 
            self.opencl.queue, ( self.neuron_count * 64, ),
            self.context.inputs_buf, self.context.weights_buf,
            numpy.int32( self.inputs_offset ),
            numpy.int32( self.weights_offset ),
            numpy.int32( self.neurons_offset ),
            numpy.int32( self.inputs_per_neuron ),
            numpy.int32( self.neuron_count ),
            pyopencl.LocalMemory( 256 ),
            self.context.outputs_buf,
            local_size = ( 64, )
            )

        self.processed = True

        for l in self.next_layers:
            l[0].process()

    def calc_weights_gradient( self ):
        """
        Calculate gradient of weights.
        
        This method should be called only for processed layers as it's used
        inputs array which is valid only at processing time.
        """

        for l in self.next_layers:
            if not l[0].processed:
                l[0].calc_weights_gradient()

#        err = numpy.ndarray( [ self.context.total_neurons ], numpy.float32 )
#        grad = numpy.ndarray( [ self.context.total_weights ], numpy.float32 )

        self.opencl.kernel_calc_layer_gradient( 
            self.opencl.queue, ( self.weights_count, ),
            self.context.inputs_buf, self.context.errors_backpropagation_buf,
            numpy.int32( self.inputs_offset ),
            numpy.int32( self.neurons_offset ),
            numpy.int32( self.inputs_per_neuron ),
            numpy.int32( self.weights_offset ),
            self.context.gradient_buf
            )

#        pyopencl.enqueue_read_buffer( self.opencl.queue, self.context.errors_backpropagation_buf, err, is_blocking = True )
#        pyopencl.enqueue_read_buffer( self.opencl.queue, self.context.gradient_buf, grad, is_blocking = True )

        i_s = numpy.int32( 1 )
        for l in self.prev_layers:
            self.opencl.kernel_propagate_errors( 
                self.opencl.queue, ( l[2] * 64, ),
                self.context.errors_backpropagation_buf,
                self.context.weights_buf,
                numpy.int32( self.neurons_offset ),
                numpy.int32( l[0].neurons_offset + l[1] ),
                numpy.int32( l[2] ),
                numpy.int32( self.neuron_count ),
                self.weights_offset + i_s,
                numpy.int32( self.inputs_per_neuron ),
                pyopencl.LocalMemory( 256 ),
                self.context.outputs_buf,
                self.context.errors_backpropagation_buf,
                local_size = ( 64, )
            )
            i_s += l[2]

        self.processed = True

class InputLayer( Layer ):
    """
    Special layer for input layer. All inputs are passed directly to linked layers.
    User must specify inputs for entire neural network by directly assigning them to
    'outputs' array on InputLayer. Then call 'process'.
    """

    def __init__( self, neuron_count, opencl ):
        """
        Increase inputs per neuron by input neurons
        """
        super( InputLayer, self ).__init__( neuron_count, opencl )

        self.inputs_per_neuron += neuron_count

    def set_inputs( self, inputs, is_blocking = True ):
        """
        Setup inputs to input layer.
        
        @param inputs
            NumPy.NDArray of float32 values, size equals to neuron count
        """
        pyopencl.enqueue_write_buffer( 
            self.opencl.queue, self.context.inputs_buf, inputs,
            device_offset = self.inputs_offset * 4, is_blocking = is_blocking
            )

    def process( self ):
        """
        Process for InputLayer does nothing. Simple invokes process for next layers.
        """
        super( InputLayer, self ).process()

        self.reset_processed()

    def calc_weights_gradient( self ):
        """
        Does nothing. Calls calc_weights_gradient on following layers.
        """
        super( InputLayer, self ).calc_weights_gradient()

        self.reset_processed()

class OutputLayer( Layer ):
    """
    Special layer for outputs.
    """

#    def setup_training_data( self, data_to_train ):
#        """
#        Setup data to train neural network to.
#        
#        @param data_to_train
#            Numpy array, size should exactly match neurons count of OutputLayer.
#        """
#        pyopencl.enqueue_write_buffer( self.opencl.queue, self.errors_backpropagation_buf, data_to_train, is_blocking = True )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
