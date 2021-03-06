"""
Created on 16.05.2010

@author: RevEn

This module defines common training methods for neural networks.

Example:
Assume we have a simple neural network with two inputs and one output
and we'd like to train it XOR function.

    Initialize neural network with 2 inputs, one hidden layers with 2 neurons
    and one output layer with one neuron
>>> from opencl import *
>>> from layer import *
>>> ocl = OpenCL( pyopencl.create_some_context( ) )
>>> i = InputLayer( 2, ocl )
>>> h = Layer( 2, ocl )
>>> o = OutputLayer( 1, ocl )
>>> i.link_next( h, 0, 2 )
>>> h.link_next( o, 0, 2 )
>>> nnc = ExecutionContext( i, o, allow_training = True )

    Example of usage TrainingResults to store and restore neural network weights
>>> tr = TrainingResults( )
>>> h.set_weights( numpy.array( [ 1, 1, 1, 1, 1, 1 ], dtype=numpy.float32 ) )
>>> tr.store_weights( nnc )
>>> old_weights = tr.optimal_weights
>>> h.get_weights( )
array([ 1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)
>>> h.set_weights( numpy.array( [ 2, 2, 2, 2, 2, 2 ], dtype=numpy.float32 ) )
>>> h.get_weights( )
array([ 2.,  2.,  2.,  2.,  2.,  2.], dtype=float32)
>>> tr.apply_weights( nnc )
>>> h.get_weights( )
array([ 1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

    Example of training neural network by gradiend descent method
>>> tr.reset( )
>>> m = GradientDescent( n = 0.5, alpha = 0.3 )
>>> training_data = (
... ( numpy.array( ( 0.0, 0.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
... ( numpy.array( ( 0.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
... ( numpy.array( ( 1.0, 0.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
... ( numpy.array( ( 1.0, 1.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
... )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.iterations
10
>>> tr.minimal_error
0.5062440037727356

    Example of training neural network by conjugate gradient method
>>> tr.reset( )
>>> m = ConjugateGradient( n = 0.5, alpha = 0.3, offline = True )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
0.50631242990493774

    Example of training neural network by Quickprop method
>>> tr.reset( )
>>> m = Quickprop( n = 0.5, alpha = 0.3, offline = True )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
0.5033717155456543
         
    Example of training neural network by RPROP method
>>> tr.reset( )
>>> m = RPROP( n = 0.5, offline = True )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
0.015365475788712502
"""

import numpy
import pyopencl #@UnresolvedImport
import time



class TrainingResults( object ):
    """
    Holds training information between training sessions
    
    Can store and restore weights on entire neural network.
    Can be pickled.
    """

    def __init__( self ):
        """
        Constructs empty results structure.
        """
        self.reset()

    def reset( self ):
        """
        Reset results.
        """
        self.iterations = numpy.int32( 0 )
        self.minimal_error = numpy.float32( 1e12 )
        self.optimal_weights = None
        self.total_time = 0.0
        self.opencl_time = 0.0

    def _iterate_over_layers( self, context, func ):
        """
        Helper method to iterate over all layers in neural
        network and apply some functor to layer
        """
        ll = [ context.input_layer ]
        while ll:
            l = ll.pop()
            l._processed = True
            func( l )
            for k in l._next_layers:
                if k[0].processed == False:
                    ll.append( k[0] )

        context.input_layer.reset_processed()

    def store_weights( self, context ):
        """
        Stores list of layers weights for entire neural network.
        """

        # iterate over all layers and apply weights to them
        self.optimal_weights = []
        def _store_weights( l ):
            self.optimal_weights.append( l.get_weights() )

        self._iterate_over_layers( context, _store_weights )

    def apply_weights( self, context ):
        """
        Apply optimal weights to neural network.
        """
        if not isinstance( self.optimal_weights, list ):
            #old format, convert optimal_weights to list

            class _gather_weights():
                def __init__( self, w ):
                    self.ofs = 0
                    self.optimal_weights = w
                    self.new_weights = []
                def __call__( self, l ):
                    self.new_weights.append( self.optimal_weights[self.ofs:self.ofs + l.weights_count] )
                    self.ofs += 16 * ( 1 + l.weights_count // 16 )   # old style alignment...

            g = _gather_weights( self.optimal_weights )
            self._iterate_over_layers( context, g )
            self.optimal_weights = g.new_weights

        # iterate over all layers and apply weights to them
        class _apply_weights():
            def __init__( self, w ):
                self.i = 0
                self.optimal_weights = w
            def __call__( self, l ):
                l.set_weights( self.optimal_weights[ self.i ] )
                self.i += 1

        self._iterate_over_layers( context, _apply_weights( self.optimal_weights ) )



class TrainingMethod( object ):
    """
    Base class for all neural network training methods.
    
    Can be pickled.
    """

    def __init__( self, n = 0.5, alpha = 0.2, kw = 1.03, pd = 0.7, pi = 1.02, offline = False ):
        """
        Construct base training method object.

        @param n
            Training coefficient (learning rate) [0, +inf).
            
        @param alpha
            Momentum coefficient. Useful when the gradient is near to zero vector.
            
        @param kw
            Acceptable increase in error function (relative).
        
        @param pd
            Auto decrease of training coefficient (relative).
            
        @param pi
            Auto increase of training coefficient (relative).
            
        @param offline
            Indicates that weights adjustment will be made after computing mean error on all
            training data.
        """
        self.n = numpy.float32( n )
        self.alpha = numpy.float32( alpha )
        self.kw = numpy.float32( kw )
        self.pd = numpy.float32( pd )
        self.pi = numpy.float32( pi )
        self.last_error = numpy.float32( 0.0 )
        self.offline = offline

    def randomize_weights( self, context ):
        """
        Initialize weights of layer by random values
        """

        weights = numpy.random.rand( context._weights_buf_size ).astype( numpy.float32 )
        weights -= 0.5
        weights *= 4.0 / numpy.sqrt( numpy.float32( context._weights_buf_size / context._neurons_buf_size ) )

        pyopencl.enqueue_write_buffer( context.opencl.queue, context._weights_buf, weights, is_blocking = True )

    def adjust_training_parameters( self, error ):
        """
        Adjust training parameters by total calculated error value
        """
        if error > self.kw * self.last_error:
            self.n *= self.pd
        elif self.n < 1.0:
            self.n *= self.pi
        self.last_error = error

    def __getstate__( self ):
        odict = self.__dict__.copy()
        del odict['_weights_delta_buf']
        return odict

    def prepare_training( self, context ):
        """
        Initialize training method by neural network.
        
        @param context
            Execution context.
        """

        self._weights_delta_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [ context._weights_buf_size ], numpy.float32 )
            )

    def start_training( self, context, training_data, training_results,
                        maximal_iterations = 10000, target_error = 0.01,
                        report = False ):
        """
        Starts training.
        
        @param context
            Input layer of neural network.
            
        @param training_data
            Array of tuples of inputs and outputs.
            
        @param training_results
            TrainingResults structure where optimal results will be stored.
            
        @param maximal_iterations
            Maximal iteration to perform.
            
        @param target_error
            Target absolute error.
            
        @param report
            Report object (optimal)
            
        @return Tuple of performed iterations count, minimal relative error
        """

        start_time = time.clock()

        self.prepare_training( context )

        total_error = numpy.array( [1e12], numpy.float32 )
        total_error_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = total_error )

        zeros_buf = pyopencl.Buffer( 
            context.opencl.context,
            pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [context._weights_buf_size], numpy.float32 )
            )

        read_ready_event = None

        o_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [context.output_layer.neuron_count], numpy.float32 )
            )

        context.opencl.kernel_setup_training_data.set_arg( 0, context._neurons_buf_size )
        context.opencl.kernel_setup_training_data.set_arg( 1, context._outputs_buf )
        context.opencl.kernel_setup_training_data.set_arg( 2, context.output_layer._neurons_offset )
        context.opencl.kernel_setup_training_data.set_arg( 3, context.output_layer.neuron_count )
        context.opencl.kernel_setup_training_data.set_arg( 4, o_buf )
        context.opencl.kernel_setup_training_data.set_arg( 5, pyopencl.LocalMemory( 32 * 4 ) )
        context.opencl.kernel_setup_training_data.set_arg( 6, context._errors_backpropagation_buf )
        context.opencl.kernel_setup_training_data.set_arg( 7, total_error_buf )

        # clear gradient
        pyopencl.enqueue_copy_buffer( 
            context.opencl.queue, zeros_buf, context._gradient_buf
            ).wait()

        i = 0
        calc_error_evt = None
        while training_results.minimal_error > target_error:
            if i >= maximal_iterations:
                break
            i += 1

            reset_total_error_evt = pyopencl.enqueue_copy_buffer( context.opencl.queue, zeros_buf, total_error_buf, byte_count = 4 )
            j = 0
            for inputs, outputs in training_data:
                j += 1
#                pyopencl.enqueue_barrier( context.opencl.queue )
                evt = context.input_layer.set_inputs( inputs, is_blocking = False )
                context.input_layer._process_wait_for.append( evt )
                context.input_layer.process()

                evt = pyopencl.enqueue_write_buffer( 
                    context.opencl.queue, o_buf, outputs, is_blocking = False
                    )

                calc_error_evt = pyopencl.enqueue_nd_range_kernel( 
                    context.opencl.queue,
                    context.opencl.kernel_setup_training_data,
                    ( 32, ), ( 32, ),
                    wait_for = ( evt, context.output_layer._process_event, reset_total_error_evt )
                    )

#                print context.output_layer.get_outputs()

                context.output_layer._calc_gradient_wait_for.append( calc_error_evt )
                context.input_layer.calc_weights_gradient()
                #print context.output_layer._get_gradient( )

                if not self.offline:
                    self.adjust_weights( context )
                    evt = pyopencl.enqueue_copy_buffer( 
                        context.opencl.queue, zeros_buf, context._gradient_buf,
                        wait_for = ( context.input_layer._calc_gradient_event, )
                        )
                    context.output_layer._calc_gradient_wait_for.append( evt )

                if j % 20000 == 0:
                    context.opencl.queue.finish()

            if self.offline:
                save_n = self.n
                self.n /= numpy.float32( len( training_data ) )
                self.adjust_weights( context )
                self.n = save_n
                evt = pyopencl.enqueue_copy_buffer( context.opencl.queue, zeros_buf, context._gradient_buf )
                context.output_layer._calc_gradient_wait_for.append( evt )

            if read_ready_event and read_ready_event.command_execution_status == pyopencl.command_execution_status.COMPLETE:
                read_ready_event = None
                error_sum = total_error[0] / len( training_data )
#                print error_sum, ' ', i, ' ', self.n

                if report:
                    report.process_iteration( len( training_data ), self, training_results, error_sum, context )

                self.adjust_training_parameters( error_sum )

                if error_sum < training_results.minimal_error:
                    training_results.minimal_error = error_sum
                    training_results.store_weights( context )   # note: this call is blocking!

                if error_sum < target_error:
                    break;

                training_results.opencl_time += context.opencl.gather_opencl_stats()

            if not read_ready_event:
                # we use nonblocking read to avoid waiting for GPU
                # this could lead to a delay in obtaining current error
                # error of current iteration can be returned in several iteration ahead
                read_ready_event = pyopencl.enqueue_read_buffer( 
                    context.opencl.queue, total_error_buf,
                    total_error, is_blocking = False,
                    wait_for = ( calc_error_evt, ) if calc_error_evt else None
                    )

        training_results.iterations += i

        pyopencl.enqueue_read_buffer( 
            context.opencl.queue, total_error_buf,
            total_error, is_blocking = True,
            wait_for = ( calc_error_evt, ) if calc_error_evt else None
            )
        error_sum = total_error[0] / len( training_data )

        if error_sum < training_results.minimal_error:
            training_results.minimal_error = error_sum
            training_results.store_weights( context )

        training_results.opencl_time += context.opencl.gather_opencl_stats()
        training_results.total_time += time.clock() - start_time

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        dir = self.get_weights_direction_buf( context ) #this call should always return opposite direction

        context.opencl.kernel_adjust_weights( 
            context.opencl.queue, ( int( context._weights_buf_size ), ),
            dir,
            self.n, self.alpha,
            self._weights_delta_buf,
            context._weights_buf,
            wait_for = ( context.input_layer._calc_gradient_event, ),
            )



class GradientDescent( TrainingMethod ):
    """
    Gradient descent optimization method. Uses gradient as a vector of weights adjustment.
    """

    def __init__( self, *karg, **kwarg ):
        """
        Constructs gradient descent training method.
        """
        super( GradientDescent, self ).__init__( *karg, **kwarg )

    def get_weights_direction_buf( self, context ):
        """
        Returns direction is weights space by which weights should be modified.
        In gradient descent method this is simply gradients vector
        """
        return context._gradient_buf



class ConjugateGradient( TrainingMethod ):
    """
    Conjugate gradients training algorithm.
    """
    def __init__( self, *karg, **kwarg ):
        """
        Constructor.
        """
        super( ConjugateGradient, self ).__init__( *karg, **kwarg )

    def prepare_training( self, context ):
        """
        Create additional buffers to store previous gradient vector.
        
        @param layer
            Input layer.
        """
        super( ConjugateGradient, self ).prepare_training( context )

        self.direction_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [ context._weights_buf_size ], numpy.float32 )
            )
        self.prev_gradient_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.01 ] * context._weights_buf_size, numpy.float32 )
            )

        #1 float beta coefficient
        self.beta_buf = pyopencl.Buffer( context.opencl.context, pyopencl.mem_flags.READ_WRITE, 4 )

        self.iteration_count = 0

    def __getstate__( self ):
        odict = super( ConjugateGradient, self ).__getstate__()
        del odict['direction_buf']
        del odict['prev_gradient_buf']
        del odict['beta_buf']
        return odict

    def get_weights_direction_buf( self, context ):
        """
        Returns direction by which adjust weights.
        """
        context.opencl.kernel_calc_conjugate_gradient_beta( 
            context.opencl.queue, ( 64, ),
            context._gradient_buf,
            self.prev_gradient_buf,
            numpy.int32( context._weights_buf_size ),
            pyopencl.LocalMemory( 256 ),
            pyopencl.LocalMemory( 256 ),
            self.beta_buf,
            local_size = ( 64, )
            )

#        test1 = numpy.ndarray( [ context.weights_buf_size ], numpy.float32 )
#        pyopencl.enqueue_read_buffer( context.opencl.queue, context.gradient_buf, test1, is_blocking = True )
#        test2 = numpy.ndarray( [ context.weights_buf_size ], numpy.float32 )
#        pyopencl.enqueue_read_buffer( context.opencl.queue, self.prev_gradient_buf, test2, is_blocking = True )
#
#        beta = numpy.float32( ( test1 * ( test1 - test2 ) ).sum() / ( test2 * test2 ).sum() )
#        pyopencl.enqueue_write_buffer( context.opencl.queue, self.beta_buf, numpy.array( [beta], numpy.float32 ), is_blocking = True )
#
#        test = numpy.ndarray( [ context.weights_buf_size ], numpy.float32 )
#        pyopencl.enqueue_read_buffer( context.opencl.queue, self.beta_buf, test, is_blocking = True )

        self.iteration_count += 1
        if self.iteration_count > context.total_neurons:
            pyopencl.enqueue_write_buffer( context.opencl.queue, self.beta_buf, numpy.zeros( [1], numpy.float32 ), is_blocking = True )
            self.iteration_count = 0

        context.opencl.kernel_calc_conjugate_gradient_direction( 
            context.opencl.queue, ( int( context._weights_buf_size ), ),
            context._gradient_buf,
            self.beta_buf,
            self.direction_buf,
            self.prev_gradient_buf
            )

#        pyopencl.enqueue_read_buffer( context.opencl.queue, self.direction_buf, test, is_blocking = True )

        return self.direction_buf

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        super( ConjugateGradient, self ).adjust_weights( context )


class Quickprop( TrainingMethod ):
    """
    Quickprop training method.
    """

    def __init__( self, *kargs, **kwargs ):
        """
        Constructor.
        
        Alpha (momentum coefficient) is used as alpha_max.
        """
        super( Quickprop, self ).__init__( *kargs, **kwargs )

    def prepare_training( self, context ):
        """
        Create additional buffers to store previous gradient vector.
        
        @param layer
            Input layer.
        """
        super( Quickprop, self ).prepare_training( context )

        self.prev_direction_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [ context._weights_buf_size ], numpy.float32 )
            )

    def __getstate__( self ):
        odict = super( Quickprop, self ).__getstate__()
        del odict['prev_direction_buf']
        return odict

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        context.opencl.kernel_adjust_weights_quickprop( 
            context.opencl.queue, ( int( context._weights_buf_size ), ),
            context._gradient_buf,
            self.prev_direction_buf,
            self.n, self.alpha,
            self._weights_delta_buf,
            context._weights_buf
            )

        pyopencl.enqueue_copy_buffer( context.opencl.queue, context._gradient_buf, self.prev_direction_buf )

    def adjust_training_parameters( self, error ):
        """
        Disable parameters adjustment since they interpreted differently.
        """
        pass



class RPROP( TrainingMethod ):
    """
    RPROP training method.
    """

    def __init__( self, *kargs, **kwargs ):
        """
        Constructor.
        """
        super( RPROP, self ).__init__( *kargs, **kwargs )

    def prepare_training( self, context ):
        """
        Create additional buffers to store learning rate for each weight.
        
        @param layer
            Input layer.
        """
        super( RPROP, self ).prepare_training( context )

        self.n_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ self.n ] * context._weights_buf_size, numpy.float32 )
            )
        self.prev_gradient_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [ context._weights_buf_size ], numpy.float32 )
            )

    def __getstate__( self ):
        odict = super( RPROP, self ).__getstate__()
        del odict['n_buf']
        del odict['prev_gradient_buf']
        return odict

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        context.opencl.kernel_adjust_weights_rprop( 
            context.opencl.queue, ( int( context._weights_buf_size ), ),
            context._gradient_buf,
            self.prev_gradient_buf,
            self.n_buf,
            context._weights_buf
            )

        #nn = numpy.ndarray( [context.weights_buf_size], numpy.float32 )
        #pyopencl.enqueue_read_buffer( context.opencl.queue, context.gradient_buf, nn, is_blocking = True )
        #pyopencl.enqueue_read_buffer( context.opencl.queue, self.n_buf, nn, is_blocking = True )

        pyopencl.enqueue_copy_buffer( context.opencl.queue, context._gradient_buf, self.prev_gradient_buf )

    def adjust_training_parameters( self, error ):
        """
        Disable parameters adjustment since they are interpreted differently.
        """
        pass



if __name__ == '__main__':
    import doctest #@UnresolvedImport
    doctest.testmod()

    import unittest #@UnresolvedImport

    class TrainingResultsTest( unittest.TestCase ):
        def setUp( self ):
            self.tr = TrainingResults()

            from opencl import OpenCL
            from layer import InputLayer, Layer, OutputLayer, ExecutionContext

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

        def test_store( self ):
            self.tr.reset()
            self.assertEqual( self.tr.iterations, numpy.int32( 0 ) )
            self.assertGreater( self.tr.minimal_error, numpy.float32( 1e6 ) )
            self.assertIsNone( self.tr.optimal_weights )
            self.assertAlmostEqual( self.tr.total_time, 0.0 )
            self.assertAlmostEqual( self.tr.opencl_time, 0.0 )

            self.i.set_inputs( numpy.array( [1.0, 1.0], numpy.float32 ), is_blocking = True )
            self.i.process()
            initial_result = self.o.get_outputs()

            self.tr.store_weights( self.nnc )
            self.i.set_weights( numpy.array( [ 0.4 ] * self.i.weights_count, numpy.float32 ) )
            self.i.process()

            self.assertNotEqual( initial_result, self.o.get_outputs() )

            self.tr.apply_weights( self.nnc )
            self.i.process()
            self.assertArrayEqual( initial_result , self.o.get_outputs() )

    class TrainingMethodTest( unittest.TestCase ):
        @classmethod
        def setUpClass( self ):
            from opencl import OpenCL
            from layer import InputLayer, OutputLayer, ExecutionContext

            self.ocl = OpenCL( pyopencl.create_some_context() )

            self.i = InputLayer( 2, self.ocl )
            self.o = OutputLayer( 1, self.ocl )

            self.i.link_next( self.o )

            self.nnc = ExecutionContext( self.i, self.o, allow_training = True )

            self.i.set_weights( numpy.array( [ 0.1 ] * self.i.weights_count, numpy.float32 ) )
            self.o.set_weights( numpy.array( [ 0.3 ] * self.o.weights_count, numpy.float32 ) )

            self.tr = TrainingResults()

            self._create_method()

        @classmethod
        def _create_method( self ):
            pass

        def assertArrayEqual( self, ar1, ar2 ):
            self.assertEqual( len( ar1 ), len( ar2 ) )
            for x, y in zip( numpy.array( ar1, numpy.float32 ), numpy.array( ar2, numpy.float32 ) ):
                self.assertAlmostEqual( x, y, places = 5 )

        def test_create( self ):
            self.setUpClass()

            if not getattr( self, 'method', None ):
                return

            self.assertAlmostEqual( self.method.n, 0.5 )
            self.assertAlmostEqual( self.method.alpha, 0.2 )
            self.assertAlmostEqual( self.method.kw, 1.03 )
            self.assertAlmostEqual( self.method.pd, 0.7 )
            self.assertAlmostEqual( self.method.pi, 1.02 )
            self.assertAlmostEqual( self.method.last_error, 0.0 )
            self.assertEqual( self.method.offline, False )

        def test_randomize_weights( self ):
            if not getattr( self, 'method', None ):
                return

            self.i.set_weights( numpy.array( [ 0.1 ] * self.i.weights_count, numpy.float32 ) )
            self.assertTrue( all( map( lambda x: abs( x - 0.1 ) < 0.0001, self.i.get_weights() ) ) )

            self.method.randomize_weights( self.nnc )
            w1 = self.i.get_weights()
            self.assertFalse( all( map( lambda x: abs( x - 0.1 ) < 0.0001, self.i.get_weights() ) ) )
            self.method.randomize_weights( self.nnc )
            self.assertFalse( all( map( lambda x: abs( x - 0.1 ) < 0.0001, self.i.get_weights() ) ) )
            self.assertNotAlmostEqual( ( w1 - self.i.get_weights() ).sum(), 0.0 )

        def test_adjust_weights( self ):
            if not getattr( self, 'method', None ):
                return

            self.method.last_error = numpy.float32( 1.0 )
            self.method.n = numpy.float32( 0.5 )
            self.method.kw = numpy.float32( 1.03 )
            self.method.pd = numpy.float32( 0.5 )
            self.method.pi = numpy.float32( 1.5 )

            self.method.adjust_training_parameters( 1.2 )
            self.assertAlmostEqual( self.method.n, 0.25 )
            self.assertAlmostEqual( self.method.last_error, 1.2 )

            self.method.adjust_training_parameters( 1.0 )
            self.assertAlmostEqual( self.method.n, 0.375 )
            self.assertAlmostEqual( self.method.last_error, 1.0 )

            self.method.adjust_training_parameters( 1.0 )
            self.assertAlmostEqual( self.method.n, 0.5625 )
            self.assertAlmostEqual( self.method.last_error, 1.0 )

        def test_prepare_training( self ):
            if not getattr( self, 'method', None ):
                return

            self.method.prepare_training( self.nnc )
            self.assertIsInstance( self.method._weights_delta_buf, pyopencl.Buffer )

    class GradientDescentTest( TrainingMethodTest ):
        @classmethod
        def _create_method( self ):
            self.method = GradientDescent()

        def _reset_data( self ):
            self._tr = TrainingResults()

            self.last_error = numpy.float32( 0.0 )
            self.method.n = numpy.float32( 0.5 )
            self.method.kw = numpy.float32( 1.03 )
            self.method.pd = numpy.float32( 0.5 )
            self.method.pi = numpy.float32( 1.5 )
            self.method.alpha = numpy.float32( 0.2 )

            self.i.set_weights( numpy.array( [ 0.1 ] * self.i.weights_count, numpy.float32 ) )
            self.o.set_weights( numpy.array( [ 0.3 ] * self.o.weights_count, numpy.float32 ) )

            self._training_data = ( 
                ( numpy.array( ( 0.0, 0.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
                ( numpy.array( ( 0.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
                ( numpy.array( ( 1.0, 0.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
                ( numpy.array( ( 1.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
            )

        def test_one_data( self ):
            self._reset_data()

            old_w = ( self.i.get_weights(), self.o.get_weights() )

            self.i.set_inputs( self._training_data[0][0] )
            self.i.process()

            total_error_buf = pyopencl.Buffer( 
                self.nnc.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
                hostbuf = numpy.array( [1e12], numpy.float32 ) )

            self.o._set_outputs_and_calc_errors( self._training_data[0][1], total_error_buf )
            self.i.calc_weights_gradient()

            grad = ( self.i._get_gradient(), self.o._get_gradient() )

            self.method.start_training( self.nnc, self._training_data[:1], self._tr, 1 )
            self.assertEqual( self._tr.iterations, 1 )
            self.assertAlmostEqual( self.o.get_outputs()[ 0 ], 0.222823, places = 5 )
            self.assertAlmostEqual( self.o._get_errors()[ 0 ], 0.141171, places = 5 )
            self.assertAlmostEqual( self._tr.minimal_error, 0.222823, places = 5 )
            self.assertArrayEqual( self.o._get_gradient(), [ 0.0, 0.0, 0.0 ] )  # gradient is cleared, it's ok

            self.assertArrayEqual( self.i.get_weights(), old_w[ 0 ] - 0.5 * grad[ 0 ] )
            self.assertArrayEqual( self.o.get_weights(), old_w[ 1 ] - 0.5 * grad[ 1 ] )

        def test_multiple_data( self ):
            self._reset_data()

            self.method.start_training( self.nnc, self._training_data, self._tr, 1 )
            err1 = self._tr.minimal_error
            self.method.start_training( self.nnc, self._training_data, self._tr, 1 )

            self.assertLess( self._tr.minimal_error, err1 )

        def test_multiple_iterations( self ):
            self._reset_data()

            self.method.alpha = numpy.float32( 0.0 )
            for i in range( 0, 5 ):
                self.method.start_training( self.nnc, self._training_data, self._tr, 1 )

            save_tr = self._tr

            self._reset_data()
            self.method.alpha = numpy.float32( 0.0 )
            self.method.start_training( self.nnc, self._training_data, self._tr, 5 )

            self.assertAlmostEqual( self._tr.minimal_error, save_tr.minimal_error )
            self.assertEqual( self._tr.iterations, save_tr.iterations )
            self.assertArrayEqual( self._tr.optimal_weights[0], save_tr.optimal_weights[0] )
            self.assertArrayEqual( self._tr.optimal_weights[1], save_tr.optimal_weights[1] )

        def test_learning( self ):
            self._reset_data()

            self.method.start_training( self.nnc, self._training_data, self._tr, 100 )
            for i, o in self._training_data:
                self.i.set_inputs( i )
                self.i.process()
                self.assertAlmostEqual( self.o.get_outputs()[0], o[0], delta = 0.1 )

    unittest.main()
