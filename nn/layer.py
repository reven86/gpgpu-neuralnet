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
>>> i.finilize_links()

    Setting up inputs for entire neural network
>>> i.set_inputs( numpy.array( ( 1, 2 ), numpy.float32 ) )

    Setting up weights of links.
>>> i.set_weights( numpy.array( ( 0, 1, 0, 0, 0, 1 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( 0, 1, 2, 3, 4, -5, 6, 7, 8 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 9, 10, 0.1, 0.2 ), numpy.float32 ) )

    Start simulation
>>> i.process()
>>> i.get_outputs( )
array([ 1.,  2.], dtype=float32)
>>> h.get_outputs( )
array([ 0.9974578 , -0.96402615,  1.        ], dtype=float32)
>>> o.get_outputs( )
array([-0.09323524], dtype=float32)

>>> o.setup_training_data( numpy.array( ( 1, ), numpy.float32 ) )
>>> i.calc_weights_gradient( )

"""

import numpy
import pyopencl

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
        self.outputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.neuron_count * 4 )
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

    def finilize_links( self ):
        """
        Creates all necessary information about linked layers.
        
        Creates OpenCL buffers, programs, etc. Must be called prior to process.
        """
        # inputs_buf doesn't store polarization link
        self.inputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_ONLY, ( max( 1, self.inputs_per_neuron - 1 ) ) * 4 )
        self.weights_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.inputs_per_neuron * self.neuron_count * 4 )

        self.gradient_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.inputs_per_neuron * self.neuron_count * 4 )
        self.errors_backpropagation_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, self.neuron_count * 4 )

        self.processed = False
        for l in self.next_layers:
            l[0].finilize_links()

    def set_weights( self, weights ):
        """
        Set weights for entire layer.
        
        @param weights
            NumPy.NDArray of float32 values, size equals to inputs_per_neuron * neuron_count
        """
        pyopencl.enqueue_write_buffer( self.opencl.queue, self.weights_buf, weights, is_blocking = True )

    def get_weights( self ):
        """
        Returns weights.
        """
        weights = numpy.ndarray( [ self.neuron_count * self.inputs_per_neuron ], numpy.float32 )
        pyopencl.enqueue_read_buffer( self.opencl.queue, self.weights_buf, weights, is_blocking = True )
        return weights

    def get_outputs( self ):
        """
        Wait for outputs.
        """
        outputs = numpy.ndarray( [ self.neuron_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( self.opencl.queue, self.outputs_buf, outputs, is_blocking = True )
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
                self.opencl.queue, l[0].outputs_buf, self.inputs_buf,
                byte_count = l[2] * 4,
                src_offset = l[1] * 4,
                dst_offset = i_s * 4
                )
            i_s += l[2]

        #process layer
        self.opencl.kernel_process_layer( 
            self.opencl.queue, ( self.neuron_count * 64, ),
            self.inputs_buf, self.weights_buf,
            numpy.int32( self.inputs_per_neuron ),
            numpy.int32( self.neuron_count ),
            pyopencl.LocalMemory( 256 ),
            self.outputs_buf,
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

        #err = numpy.ndarray( [ self.neuron_count ], numpy.float32 )
        #grad = numpy.ndarray( [ self.neuron_count * self.inputs_per_neuron ], numpy.float32 )

        self.opencl.kernel_calc_layer_gradient( 
            self.opencl.queue, ( self.neuron_count * self.inputs_per_neuron, ),
            self.inputs_buf, self.errors_backpropagation_buf,
            numpy.int32( self.inputs_per_neuron ),
            self.gradient_buf
            )

        #pyopencl.enqueue_read_buffer( self.opencl.queue, self.errors_backpropagation_buf, err )
        #pyopencl.enqueue_read_buffer( self.opencl.queue, self.gradient_buf, grad )

        i_s = numpy.int32( 1 )
        for l in self.prev_layers:
            self.opencl.kernel_propagate_errors( 
                self.opencl.queue, ( l[2] * 64, ),
                self.errors_backpropagation_buf,
                self.weights_buf,
                numpy.int32( l[1] ),
                numpy.int32( l[2] ),
                numpy.int32( self.neuron_count ),
                i_s,
                numpy.int32( self.inputs_per_neuron ),
                pyopencl.LocalMemory( 256 ),
                l[0].outputs_buf,
                l[0].errors_backpropagation_buf,
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
        pyopencl.enqueue_write_buffer( self.opencl.queue, self.inputs_buf, inputs, is_blocking = is_blocking )

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
