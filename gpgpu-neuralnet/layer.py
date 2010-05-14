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
      \/  H  x - O
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
>>> i.outputs[:] = ( 1, 2 )
>>> i.outputs
array([ 1.,  2.], dtype=float32)

    Setting up weights of links.
>>> h.weights[:]=( 0, 1, 2, 3, 4, -5, 6, 7, 8 )
>>> o.weights[:]=( 9, 10, 0.1, 0.2 )
>>> h.weights
array([ 0.,  1.,  2.,  3.,  4., -5.,  6.,  7.,  8.], dtype=float32)
>>> o.weights
array([  9. ,  10. ,   0.1,   0.2], dtype=float32)

    Start simulation
>>> i.process()
>>> o.outputs
array([-0.09323524], dtype=float32)

"""

import numpy
import pyopencl
import opencl

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
        self.outputs = numpy.ndarray( [neuron_count], numpy.float32 )
        self.outputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.WRITE_ONLY, self.outputs.nbytes )
        self.prev_layers = []
        self.next_layers = []
        self.processed = False
        self.inputs_per_neuron = numpy.int32( 1 )      # polarization input is always exists

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
        self.inputs = numpy.ndarray( [ self.inputs_per_neuron ], numpy.float32 )
        self.weights = numpy.ndarray( [ self.inputs_per_neuron * self.neuron_count ], numpy.float32 )
        self.inputs[0] = numpy.float32( 1.0 )

        self.inputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.USE_HOST_PTR, hostbuf = self.inputs )
        self.weights_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.USE_HOST_PTR, hostbuf = self.weights )

        self.processed = False

        for l in self.next_layers:
            l[0].finilize_links()

    def get_outputs( self ):
        """
        Wait for outputs.
        """
        self.process_wait_event.wait()

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
        Use get_outputs to wait for GPU to completion if computations.
        Must be called only if layer has a link to next layer.
        """

        # ensure that all previous layers are processed
        for l in self.prev_layers:
            if not l[0].processed:
                return

        i_s = 1
        for l in self.prev_layers:
            l[0].get_outputs()
            self.inputs[i_s:i_s + l[2]] = l[0].outputs[l[1]:l[1] + l[2]]
            i_s += l[2]

        #process layer
        self.opencl.program.process_layer( 
            self.opencl.queue, self.outputs.shape,
            self.inputs_buf, self.weights_buf,
            self.inputs_per_neuron,
            self.outputs_buf )

        self.process_wait_event = pyopencl.enqueue_read_buffer( self.opencl.queue, self.outputs_buf, self.outputs )

        self.processed = True

        for l in self.next_layers:
            l[0].process()

class InputLayer( Layer ):
    """
    Special layer for input layer. All inputs are passed directly to linked layers.
    User must specify inputs for entire neural network by directly assigning them to
    'outputs' array on InputLayer. Then call 'process'.
    """

    def get_outputs( self ):
        """
        Do nothing since outputs are simply passed to next layer.
        """

    def process( self ):
        """
        Process for InputLayer does nothing. Simple invokes process for next layers.
        """
        self.processed = True

        for l in self.next_layers:
            l[0].process()

        #immediately reset 'processed' flag as we process entire network
        self.reset_processed()

class OutputLayer( Layer ):
    """
    Special layer for outputs.
    """
    def process( self ):
        """
        Automatically calls get_outputs since there is no calculations following
        output layer.
        """
        super( OutputLayer, self ).process()
        self.get_outputs()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
