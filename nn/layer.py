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
<pyopencl._cl.Event object at ...>
 
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
import pyopencl #@UnresolvedImport

class ExecutionContext( object ):
    """
    Holds necessary data (buffers, kernels) for entire neural network that are used
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
        self._opencl = input_layer.opencl
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._training_allowed = allow_training
        self._total_neurons = numpy.int32( 0 )
        self._total_weights = numpy.int32( 0 )
        self._total_inputs = numpy.int32( 0 )      # total inputs to neurons, without polarization link

        # following variables define actual items count in buffers and may differ from
        # total_* variables due to alignment issues
        self._neurons_buf_size = numpy.int32( 0 )
        self._weights_buf_size = numpy.int32( 0 )
        self._inputs_buf_size = numpy.int32( 0 )

        def align( value, threshold ):
            return ( value + threshold - 1 ) & ~( threshold - 1 )

        ll = [ input_layer ]
        while ll:
            l = ll.pop()

            #process layer
            l._weights_count = l.neuron_count * l.inputs_per_neuron
            l._weights_offset = self._weights_buf_size
            l._neurons_offset = self._neurons_buf_size
            l._inputs_offset = self._inputs_buf_size
            l.context = self
            l._processed = True

            self._total_weights += l.weights_count
            self._total_neurons += l.neuron_count
            self._total_inputs += l.inputs_per_neuron - 1

            l._neurons_buf_size = align( l.neuron_count, 32 )
            l._weights_buf_size = align( l.weights_count, self.opencl.max_local_size[ 0 ] )
            l._inputs_buf_size = align( l.inputs_per_neuron - 1, 16 )

            self._neurons_buf_size += l._neurons_buf_size
            self._weights_buf_size += l._weights_buf_size
            self._inputs_buf_size += l._inputs_buf_size

            for k in l._next_layers:
                if k[0].processed == False:
                    ll.append( k[0] )

        input_layer.reset_processed()

        if allow_training:
            fl = pyopencl.mem_flags.READ_WRITE
        else:
            fl = pyopencl.mem_flags.READ_ONLY

        self._inputs_buf = pyopencl.Buffer( 
            self.opencl.context,
            pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [self._inputs_buf_size], numpy.float32 )
            )
        self._outputs_buf = pyopencl.Buffer( 
            self.opencl.context,
            pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [self._neurons_buf_size], numpy.float32 )
            )
        self._weights_buf = pyopencl.Buffer( 
            self.opencl.context, fl | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [self._weights_buf_size], numpy.float32 )
            )

        if allow_training:
            self._gradient_buf = pyopencl.Buffer( 
                self.opencl.context,
                pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf = numpy.zeros( [self._weights_buf_size], numpy.float32 )
                )
            self._errors_backpropagation_buf = pyopencl.Buffer( 
                self.opencl.context,
                pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf = numpy.zeros( [self._neurons_buf_size], numpy.float32 )
                )

    @property
    def opencl( self ):
        return self._opencl

    @property
    def input_layer( self ):
        return self._input_layer

    @property
    def output_layer( self ):
        return self._output_layer

    @property
    def training_allowed( self ):
        return self._training_allowed

    @property
    def total_neurons( self ):
        return self._total_neurons

    @property
    def total_inputs( self ):
        return self._total_inputs

    @property
    def total_weights( self ):
        return self._total_weights



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

        self._neuron_count = numpy.int32( neuron_count )
        self.opencl = opencl
        self._prev_layers = []
        self._next_layers = []
        self._processed = False
        self._inputs_per_neuron = numpy.int32( 1 )     # polarization input is always exists
        self._process_event = None
        self._process_wait_for = []
        self._calc_gradient_event = None
        self._calc_gradient_wait_for = []

    @property
    def neuron_count( self ):
        return self._neuron_count

    @property
    def processed( self ):
        return self._processed

    @property
    def inputs_per_neuron( self ):
        return self._inputs_per_neuron

    @property
    def weights_count( self ):
        return self._weights_count

    def link_next( self, next_layer, this_start_neuron = 0, this_neurons_count = -1 ):
        """
        Links this layer with next layer.
        
        @param next_layer
            Layer to link this layer with. Part of neurons from this layer
            will be linked with all neurons from next layer, each with each.
        @param this_start_neuron, this_neurons_count
            Range of neurons that will be linked with next layer.
        """

        nc = numpy.int32( this_neurons_count )
        if nc < 0:
            nc = self.neuron_count

        self._next_layers.append( ( next_layer, numpy.int32( this_start_neuron ), nc ) )
        next_layer._prev_layers.append( ( self, numpy.int32( this_start_neuron ), nc ) )
        next_layer._inputs_per_neuron += nc

    def set_weights( self, weights ):
        """
        Set weights for entire layer.
        
        @param weights
            NumPy.NDArray of float32 values, size equals to inputs_per_neuron * neuron_count
        """
        pyopencl.enqueue_write_buffer( 
            self.opencl.queue, self.context._weights_buf, weights,
            device_offset = int( self._weights_offset * 4 ), is_blocking = True
            )

    def get_weights( self ):
        """
        Returns weights.
        """
        weights = numpy.ndarray( [ self._weights_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context._weights_buf, weights,
            device_offset = int( self._weights_offset * 4 ), is_blocking = True
            )
        return weights

    def get_inputs( self ):
        """
        Returns inputs.
        """
        inputs = numpy.ndarray( [ self.inputs_per_neuron - 1 ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context._inputs_buf, inputs,
            device_offset = int( self._inputs_offset * 4 ), is_blocking = True
            )
        return inputs

    def get_outputs( self ):
        """
        Wait for outputs.
        """
        outputs = numpy.ndarray( [ self.neuron_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context._outputs_buf, outputs,
            device_offset = int( self._neurons_offset * 4 ), is_blocking = True
            )
        return outputs

    def _get_gradient( self ):
        """
        Gets weights gradient vector of this layer.
        """
        gradient = numpy.ndarray( [ self.weights_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context._gradient_buf, gradient,
            device_offset = int( self._weights_offset * 4 ), is_blocking = True
            )
        return gradient

    def _get_errors( self ):
        """
        Gets propagated errors of this layer.
        """
        err = numpy.ndarray( [ self.neuron_count ], numpy.float32 )
        pyopencl.enqueue_read_buffer( 
            self.opencl.queue, self.context._errors_backpropagation_buf, err,
            device_offset = int( self._neurons_offset * 4 ), is_blocking = True
            )
        return err

    def reset_processed( self ):
        """
        Recursively reset processed flag on all linked layers.
        """
        self._processed = False
        for l in self._next_layers:
            l[0].reset_processed()

    def process( self ):
        """
        Process signal by this layer.
        
        Invokes OpenCL program that produces output array in background.
        """

        # ensure that all previous layers are processed
        for l in self._prev_layers:
            if not l[0].processed:
                return

        outbuf = self.context._outputs_buf
        inbuf = self.context._inputs_buf
        queue = self.opencl.queue

        i_s = 0
        for l in self._prev_layers:
            self._process_wait_for.append( pyopencl.enqueue_copy_buffer( 
                queue, outbuf, inbuf,
                byte_count = int( l[2] * 4 ),
                src_offset = int( ( l[0]._neurons_offset + l[1] ) * 4 ),
                dst_offset = int( ( self._inputs_offset + i_s ) * 4 ),
                wait_for = ( l[0]._process_event, )
                ) )
            i_s += l[2]

        #process layer
        kernel = self.opencl.kernel_process_layer
        kernel.set_arg( 2, self._inputs_offset )
        kernel.set_arg( 3, self._weights_offset )
        kernel.set_arg( 4, self._neurons_offset )
        kernel.set_arg( 5, self._inputs_per_neuron )
        kernel.set_arg( 6, self._neuron_count )

        self._process_event = pyopencl.enqueue_nd_range_kernel( queue, kernel,
            ( int( self._neuron_count * 64 ), ), ( 64, ),
            wait_for = self._process_wait_for
            )
        del self._process_wait_for[:]

        self._processed = True

        for l in self._next_layers:
            l[0].process()

    def calc_weights_gradient( self ):
        """
        Calculate gradient of weights.
        
        This method should be called only for processed layers as it's used
        inputs array which is valid only at processing time.
        """

        for l in self._next_layers:
            if not l[0].processed:
                l[0].calc_weights_gradient()

        queue = self.opencl.queue
        kernel = self.opencl.kernel_calc_layer_gradient

        kernel.set_arg( 2, self._inputs_offset )
        kernel.set_arg( 3, self._neurons_offset )
        kernel.set_arg( 4, self._inputs_per_neuron )
        kernel.set_arg( 5, self._weights_offset )
        kernel.set_arg( 7, self._weights_count )
        kernel.set_arg( 8, pyopencl.LocalMemory( int( 
                    4 * ( self._inputs_per_neuron + 1 + self.opencl.max_local_size[ 0 ] // self._inputs_per_neuron ) ) ) )

        self._calc_gradient_event = pyopencl.enqueue_nd_range_kernel( queue, kernel,
            ( int( self._weights_buf_size ), ), ( self.opencl.max_local_size[ 0 ], ),
            wait_for = self._calc_gradient_wait_for
            )
        del self._calc_gradient_wait_for[:]

        kernel = self.opencl.kernel_propagate_errors
        kernel.set_arg( 2, self._neurons_offset )
        kernel.set_arg( 5, self._neuron_count )
        kernel.set_arg( 7, self._inputs_per_neuron )

        i_s = numpy.int32( 1 )
        for l in self._prev_layers:
            kernel.set_arg( 3, l[0]._neurons_offset + l[1] )
            kernel.set_arg( 4, l[2] )
            kernel.set_arg( 6, self._weights_offset + i_s )

            l[0]._calc_gradient_wait_for.append( pyopencl.enqueue_nd_range_kernel( queue, kernel,
                ( int( l[2] * 64 ), ), ( 64, ),
                wait_for = ( self._calc_gradient_event, )
                ) )

            i_s += l[2]

        self._processed = True

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

        self._inputs_per_neuron += neuron_count

    def set_inputs( self, inputs, is_blocking = True, wait_for = None ):
        """
        Setup inputs to input layer.
        
        @param inputs
            NumPy.NDArray of float32 values, size equals to neuron count
        """
        return pyopencl.enqueue_write_buffer( 
            self.opencl.queue, self.context._inputs_buf, inputs,
            device_offset = int( self._inputs_offset * 4 ), is_blocking = is_blocking,
            wait_for = wait_for
            )

    def process( self ):
        """
        Process for InputLayer does nothing. Simple invokes process for next layers.
        """
        self.opencl.kernel_process_layer.set_arg( 0, self.context._inputs_buf )
        self.opencl.kernel_process_layer.set_arg( 1, self.context._weights_buf )
        self.opencl.kernel_process_layer.set_arg( 7, pyopencl.LocalMemory( 64 * 4 ) )
        self.opencl.kernel_process_layer.set_arg( 8, self.context._outputs_buf )

        if self.context.training_allowed:
            self.opencl.kernel_calc_layer_gradient.set_arg( 0, self.context._inputs_buf )
            self.opencl.kernel_calc_layer_gradient.set_arg( 1, self.context._errors_backpropagation_buf )
            self.opencl.kernel_calc_layer_gradient.set_arg( 6, self.context._gradient_buf )

            self.opencl.kernel_propagate_errors.set_arg( 0, self.context._errors_backpropagation_buf )
            self.opencl.kernel_propagate_errors.set_arg( 1, self.context._weights_buf )
            self.opencl.kernel_propagate_errors.set_arg( 8, pyopencl.LocalMemory( 256 ) )
            self.opencl.kernel_propagate_errors.set_arg( 9, self.context._outputs_buf )

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

if __name__ == '__main__':
    import doctest #@UnresolvedImport
    doctest.testmod( optionflags = doctest.ELLIPSIS )

    import unittest #@UnresolvedImport

    class LayerTest( unittest.TestCase ):
        def setUp( self ):
            from opencl import OpenCL
            self.ocl = OpenCL( pyopencl.create_some_context() )

            self.i = InputLayer( 2, self.ocl )
            self.h = Layer( 3, self.ocl )
            self.o = OutputLayer( 1, self.ocl )

            self.i.link_next( self.h )
            self.h.link_next( self.o, 0, 3 )

            self.nnc = ExecutionContext( self.i, self.o, allow_training = True )

            self.i.set_weights( numpy.array( [ 0.1 ] * self.i.weights_count, numpy.float32 ) )
            self.h.set_weights( numpy.array( [ 0.2 ] * self.h.weights_count, numpy.float32 ) )
            self.o.set_weights( numpy.array( [ 0.3 ] * self.o.weights_count, numpy.float32 ) )

        def assertArrayEqual( self, ar1, ar2 ):
            self.assertEqual( len( ar1 ), len( ar2 ) )
            for x, y in zip( numpy.array( ar1, numpy.float32 ), numpy.array( ar2, numpy.float32 ) ):
                self.assertAlmostEqual( x, y, places = 5 )

        def test_create( self ):
            self.assertEqual( self.i.neuron_count, 2 )
            self.assertEqual( self.h.neuron_count, 3 )
            self.assertEqual( self.o.neuron_count, 1 )

            self.assertEqual( self.i.weights_count, 6 )
            self.assertEqual( self.h.weights_count, 9 )
            self.assertEqual( self.o.weights_count, 4 )

            self.assertArrayEqual( self.i.get_weights(), [ 0.1 ] * self.i.weights_count )
            self.assertArrayEqual( self.h.get_weights(), [ 0.2 ] * self.h.weights_count )
            self.assertArrayEqual( self.o.get_weights(), [ 0.3 ] * self.o.weights_count )

        def test_process( self ):
            self.i.set_inputs( numpy.array( [0.0, 0.0], numpy.float32 ), is_blocking = True )
            self.i.process()

            self.assertArrayEqual( self.i.get_inputs(), [ 0.0, 0.0 ] )
            self.assertArrayEqual( self.i.get_outputs(), [ 0.066567414, 0.066567414 ] )
            self.assertArrayEqual( self.h.get_inputs(), [ 0.066567414, 0.066567414 ] )
            self.assertArrayEqual( self.h.get_outputs(), [ 0.14994399, 0.14994399, 0.14994399 ] )
            self.assertArrayEqual( self.o.get_inputs(), [ 0.14994399, 0.14994399, 0.14994399 ] )
            self.assertArrayEqual( self.o.get_outputs(), [ 0.28210124 ] )

            self.i.set_inputs( numpy.array( [0.0, 1.0], numpy.float32 ), is_blocking = True )
            self.i.process()

            self.assertArrayEqual( self.i.get_inputs(), [ 0.0, 1.0 ] )
            self.assertArrayEqual( self.i.get_outputs(), [ 0.13254748, 0.13254748 ] )
            self.assertArrayEqual( self.h.get_inputs(), [ 0.13254748, 0.13254748 ] )
            self.assertArrayEqual( self.h.get_outputs(), [ 0.16709591, 0.16709591, 0.16709591 ] )
            self.assertArrayEqual( self.o.get_inputs(), [ 0.16709591, 0.16709591, 0.16709591 ] )
            self.assertArrayEqual( self.o.get_outputs(), [ 0.29154554 ] )

        def test_gradient( self ):
            self.i.set_inputs( numpy.array( [1.0, 0.0], numpy.float32 ), is_blocking = True )
            self.i.process()

            self.assertArrayEqual( self.o.get_outputs(), [ 0.29154554 ] )

            o_buf = pyopencl.Buffer( 
                self.nnc.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf = numpy.ones( [self.nnc.output_layer.neuron_count], numpy.float32 )
                )

            total_error_buf = pyopencl.Buffer( 
                self.nnc.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf = numpy.array( [1e12], numpy.float32 ) )

            self.nnc.opencl.kernel_setup_training_data.set_args( 
                self.nnc._neurons_buf_size, self.nnc._outputs_buf, self.nnc.output_layer._neurons_offset,
                self.nnc.output_layer.neuron_count, o_buf, pyopencl.LocalMemory( 32 * 4 ),
                self.nnc._errors_backpropagation_buf, total_error_buf )

            pyopencl.enqueue_nd_range_kernel( 
                self.nnc.opencl.queue,
                self.nnc.opencl.kernel_setup_training_data,
                ( 32, ), ( 32, ),
                None, None
                ).wait()

            self.i.calc_weights_gradient()

            err = numpy.ndarray( [ self.nnc._neurons_buf_size ], numpy.float32 )
            grad = numpy.ndarray( [ self.nnc._weights_buf_size ], numpy.float32 )

            pyopencl.enqueue_read_buffer( self.ocl.queue, self.nnc._errors_backpropagation_buf, err, is_blocking = True )
            pyopencl.enqueue_read_buffer( self.ocl.queue, self.nnc._gradient_buf, grad, is_blocking = True )

            real_err = ( 0.29154554 - 1.0 ) * 0.6666666 * ( 1.0 - 0.29154554 * 0.29154554 )
            self.assertAlmostEqual( err[self.o._neurons_offset], real_err, places = 5 )
            self.assertArrayEqual( grad[self.o._weights_offset:self.o._weights_offset + self.o._weights_count], [ real_err ] + list( self.o.get_inputs() * real_err ) )
            self.assertArrayEqual( grad[self.i._weights_offset:self.i._weights_offset + self.i._weights_count], [ -0.033015892, -0.033015892, 0.0 ] * 2 )
            self.assertArrayEqual( grad[self.h._weights_offset:self.h._weights_offset + self.h._weights_count], [ -0.08401663, -0.01113619, -0.01113619 ] * 3 )

            self.assertArrayEqual( self.i._get_gradient(), [ -0.033015892, -0.033015892, 0.0 ] * 2 )

    class ComplexNNTest( unittest.TestCase ):
        def setUp( self ):
            from opencl import OpenCL
            self.ocl = OpenCL( pyopencl.create_some_context() )

            self.i = InputLayer( 10, self.ocl )
            self.h1 = Layer( 3, self.ocl )
            self.h2 = Layer( 3, self.ocl )
            self.h3 = Layer( 3, self.ocl )
            self.o = OutputLayer( 4, self.ocl )

            self.i.link_next( self.h1, 0, 3 )
            self.i.link_next( self.h2, 0, 5 )
            self.i.link_next( self.h3, 4, 6 )
            self.i.link_next( self.o, 9, 1 )
            self.h1.link_next( self.h3 )
            self.h2.link_next( self.h3, 0, 2 )
            self.h2.link_next( self.o )
            self.h3.link_next( self.o )

            self.nnc = ExecutionContext( self.i, self.o, allow_training = True )

        def assertArrayEqual( self, ar1, ar2 ):
            self.assertEqual( len( ar1 ), len( ar2 ) )
            for x, y in zip( numpy.array( ar1, numpy.float32 ), numpy.array( ar2, numpy.float32 ) ):
                self.assertAlmostEqual( x, y, places = 5 )

        def test_process( self ):
            weights = numpy.random.rand( self.nnc._weights_buf_size ).astype( numpy.float32 )
            weights -= 0.5
            weights *= 4.0 / numpy.sqrt( numpy.float32( self.nnc._weights_buf_size / self.nnc._neurons_buf_size ) )

            pyopencl.enqueue_write_buffer( self.ocl.queue, self.nnc._weights_buf, weights, is_blocking = True )

            self.i.set_inputs( numpy.array( [x * x for x in range( 0, 10 )], numpy.float32 ), is_blocking = True )
            self.i.process()

            self.assertArrayEqual( self.i.get_outputs()[:3], self.h1.get_inputs() )
            self.assertArrayEqual( self.i.get_outputs()[:5], self.h2.get_inputs() )
            self.assertArrayEqual( self.i.get_outputs()[4:10], self.h3.get_inputs()[:6] )

    unittest.main()
