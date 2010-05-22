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
>>> gd = GradientDescent( ocl, n = 0.8, alpha = 0.9 )
>>> training_data = (
... ( numpy.array( ( 0.0, 0.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
... ( numpy.array( ( 0.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
... ( numpy.array( ( 1.0, 0.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
... ( numpy.array( ( 1.0, 1.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
... )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> gd.start_training( nnc, training_data, tr, 10 )
>>> tr.iterations
11
>>> tr.minimal_error
1.4138576
>>> tr.optimal_weights
array([ -3.22724581, -10.19999981,   5.59999275,  -2.94871402,
         6.9601016 , -10.45989704,  -3.2390008 , -10.18134212,
         5.61851788,  -3.04143   ,   7.03013992, -10.39038754,
         4.84151077,   1.59103739,   3.10596204], dtype=float32)
"""

import numpy
import pyopencl

class TrainingResults( object ):
    """
    Holds training information between training sessions
    
    Can store and restore weights on entire neural network.
    """

    def __init__( self ):
        """
        Constructs empty results structure.
        """

        self.iterations = numpy.int32( 0 )
        self.minimal_error = numpy.float32( 1e12 )
        self.optimal_weights = None
        self.total_time = 0.0
        self.opencl_time = 0.0

    def store_weights( self, context ):
        """
        Stores list of layers weights for entire neural network.
        """
        self.optimal_weights = numpy.ndarray( [context.total_weights], numpy.float32 )
        pyopencl.enqueue_read_buffer( context.opencl.queue, context.weights_buf, self.optimal_weights, is_blocking = True )

    def apply_weights( self, context ):
        """
        Apply optimal weights to neural network.
        """
        pyopencl.enqueue_write_buffer( context.opencl.queue, context.weights_buf, self.optimal_weights )

class TrainingMethod( object ):
    """
    Base class for all neural network training methods.
    """

    def __init__( self, opencl, n = 0.5, alpha = 0.2, kw = 1.03, pd = 0.7, pi = 1.02 ):
        """
        Construct base training method object.

        @param n
            Training coefficient [0, 1].
            
        @param alpha
            Momentum coefficient. Useful when the gradient is near to zero vector.
            
        @param kw
            Acceptable increase in error function (relative).
        
        @param pd
            Auto decrease of training coefficient (relative).
            
        @param pi
            Auto increase of training coefficient (relative).
        """
        self.opencl = opencl
        self.n = numpy.float32( n )
        self.alpha = numpy.float32( alpha )
        self.kw = numpy.float32( kw )
        self.pd = numpy.float32( pd )
        self.pi = numpy.float32( pi )
        self.last_error = numpy.float32( 0.0 )

    def randomize_weights( self, context ):
        """
        Initialize weights of layer by random values
        """

        weights = numpy.random.rand( context.total_weights ).astype( numpy.float32 )
        weights -= 0.5
        weights *= 4.0 / numpy.sqrt( numpy.float32( context.total_weights / context.total_neurons ) )

        pyopencl.enqueue_write_buffer( context.opencl.queue, context.weights_buf, weights )

    def adjust_training_parameters( self, error ):
        """
        Adjust training parameters by total calculated error value
        """
        if error > self.kw * self.last_error:
            self.n *= self.pd
        elif self.n < 1.0:
            self.n *= self.pi
        self.last_error = error

    def prepare_training( self, context ):
        """
        Initialize training method by neural network.
        
        @param context
            Execution context.
        """

        self.weights_delta_buf = pyopencl.Buffer( 
            self.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )

    def start_training( self, context, training_data, training_results,
                        maximal_iterations = 10000, target_error = 0.01 ):
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
            
        @return Tuple of performed iterations count, minimal relative error
        """

        self.prepare_training( context )

        outputs_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_ONLY, context.output_layer.neuron_count * 4 )
        total_error_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, context.output_layer.neuron_count * 4 )

        total_error = numpy.array( [ 0.0 for x in range( context.output_layer.neuron_count ) ], numpy.float32 )

        zero_buf = pyopencl.Buffer( 
            self.opencl.context,
            pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = total_error )

        i = 0
        while training_results.minimal_error > target_error:
            i += 1
            if i > maximal_iterations:
                break

            pyopencl.enqueue_copy_buffer( self.opencl.queue, zero_buf, total_error_buf )
            for inputs, outputs in training_data:
                context.input_layer.set_inputs( inputs, False )
                context.input_layer.process()

                pyopencl.enqueue_write_buffer( self.opencl.queue, outputs_buf, outputs )
                self.opencl.kernel_setup_training_data( 
                    self.opencl.queue, ( context.output_layer.neuron_count, ),
                    context.outputs_buf,
                    numpy.int32( context.output_layer.neurons_offset ),
                    outputs_buf,
                    context.errors_backpropagation_buf,
                    total_error_buf
                    )

#                print output_layer.get_outputs()

                context.input_layer.calc_weights_gradient()
                self.adjust_weights( context )
                self.complete_iteration()

            pyopencl.enqueue_read_buffer( self.opencl.queue, total_error_buf, total_error, is_blocking = True )

            error_sum = numpy.sqrt( total_error.sum() )

            self.adjust_training_parameters( error_sum )

            if error_sum < training_results.minimal_error:
                training_results.minimal_error = error_sum
                training_results.store_weights( context )

            if error_sum < target_error:
                break;

        training_results.iterations += i

    def complete_iteration( self ):
        """
        Complete training iteration. Does nothing for mostly of training methods.
        """
        pass

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        dir = self.get_weights_direction_buf( context ) #this call should always return opposite direction

        self.opencl.kernel_adjust_weights( 
            self.opencl.queue, ( context.total_weights, ),
            dir,
            self.n, self.alpha,
            self.weights_delta_buf,
            context.weights_buf
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
        return context.gradient_buf

class ConjugateGradient( TrainingMethod ):
    """
    Conjugate gradients training algorithm.
    """
    def __init__( self, *karg, **kwarg ):
        """
        Constructor.
        """
        super( ConjugateGradient, self ).__init__( *karg, **kwarg )

        self.iterations_to_reset = 10#iterations_to_reset

        #1 float beta coefficient
        self.beta_buf = pyopencl.Buffer( self.opencl.context, pyopencl.mem_flags.READ_WRITE, 4 )

    def prepare_training( self, context ):
        """
        Create additional buffers to store previous gradient vector.
        
        @param layer
            Input layer.
        """
        super( ConjugateGradient, self ).prepare_training( context )

        self.prev_direction_buf = pyopencl.Buffer( 
            self.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )
        self.direction_buf = pyopencl.Buffer( 
            self.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )
        self.prev_gradient_buf = pyopencl.Buffer( 
            self.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.01 for x in range( context.total_weights ) ], numpy.float32 )
            )

        self.iteration_count = 0

    def get_weights_direction_buf( self, context ):
        """
        Returns direction by which adjust weights.
        """
        self.opencl.kernel_calc_conjugate_gradient_beta( 
            self.opencl.queue, ( context.total_neurons * 64, ),
            context.gradient_buf,
            self.prev_gradient_buf,
            numpy.int32( context.total_weights ),
            pyopencl.LocalMemory( 256 ),
            pyopencl.LocalMemory( 256 ),
            self.beta_buf,
            local_size = ( 64, )
            )

        test = numpy.ndarray( [ context.total_weights ], numpy.float32 )
        pyopencl.enqueue_read_buffer( self.opencl.queue, self.beta_buf, test, is_blocking = True )

        self.opencl.kernel_calc_conjugate_gradient_direction( 
            self.opencl.queue, ( context.total_weights, ),
            context.gradient_buf,
            self.prev_direction_buf,
            self.beta_buf,
            self.direction_buf,
            )

        pyopencl.enqueue_read_buffer( self.opencl.queue, self.direction_buf, test, is_blocking = True )

        pyopencl.enqueue_copy_buffer( 
            self.opencl.queue,
            context.gradient_buf,
            self.prev_gradient_buf
            )

        return self.direction_buf

    def complete_iteration( self ):
        """
        Complete training iteration. Copies 'current' buffers to previous one.
        """
        self.iteration_count += 1
        if self.iteration_count > self.iterations_to_reset:
            self.iteration_count = 0
            pyopencl.enqueue_write_buffer( self.opencl.queue, self.prev_direction_buf, numpy.zeros( [self.total_weights], numpy.float32 ) )
        else:
            pyopencl.enqueue_copy_buffer( self.opencl.queue, self.direction_buf, self.prev_direction_buf )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
