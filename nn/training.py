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
0.50626325607299805

    Example of training neural network by conjugate gradient method
>>> tr.reset( )
>>> m = ConjugateGradient( n = 0.5, alpha = 0.3, offline = True )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
0.50631535053253174

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
        del odict['weights_delta_buf']
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

        i = 0
        calc_error_evt = None
        while training_results.minimal_error > target_error:
            if i >= maximal_iterations:
                break
            i += 1

            reset_total_error_evt = pyopencl.enqueue_copy_buffer( context.opencl.queue, zeros_buf, total_error_buf, byte_count = 4 )
            for inputs, outputs in training_data:
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

                if not self.offline:
                    self.adjust_weights( context )
                    evt = pyopencl.enqueue_copy_buffer( 
                        context.opencl.queue, zeros_buf, context._gradient_buf,
                        wait_for = ( context.input_layer._calc_gradient_event, )
                        )
                    context.output_layer._calc_gradient_wait_for.append( evt )

            if self.offline:
                save_n = self.n
                self.n /= numpy.float32( len( training_data ) )
                self.adjust_weights( context )
                self.n = save_n
                evt = pyopencl.enqueue_copy_buffer( context.opencl.queue, zeros_buf, context._gradient_buf )
                context.output_layer._calc_gradient_wait_for.append( evt )

#            print read_ready_event and read_ready_event.command_execution_status
            if not read_ready_event:
                # we use nonblocking read to avoid waiting for GPU
                # this could lead to a delay in obtaining current error
                # error of current iteration can be returned in several iteration ahead
                read_ready_event = pyopencl.enqueue_read_buffer( 
                    context.opencl.queue, total_error_buf,
                    total_error, is_blocking = False,
                    wait_for = ( calc_error_evt, )
                    )

            if read_ready_event.command_execution_status == pyopencl.command_execution_status.COMPLETE:
                error_sum = total_error[0] / len( training_data )
                #print error_sum, ' ', i, ' ', self.n

                if report:
                    report.process_iteration( len( training_data ), self, training_results, error_sum, context )

                self.adjust_training_parameters( error_sum )

                if error_sum < training_results.minimal_error:
                    training_results.minimal_error = error_sum
                    training_results.store_weights( context )   # note: this call is blocking!

                if error_sum < target_error:
                    break;

                training_results.opencl_time += context.opencl.gather_opencl_stats()

        training_results.iterations += i

        pyopencl.enqueue_read_buffer( 
            context.opencl.queue, total_error_buf,
            total_error, is_blocking = True,
            wait_for = ( calc_error_evt, )
            )
        error_sum = total_error[0] / len( training_data )

        self.adjust_training_parameters( error_sum )

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
            wait_for = ( context.input_layer._calc_gradient_event, )
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

    unittest.main()
