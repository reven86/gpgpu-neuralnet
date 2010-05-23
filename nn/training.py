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
>>> m = GradientDescent( ocl, n = 0.5, alpha = 0.3 )
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
1.4140973
>>> tr.optimal_weights
array([ -3.22067547, -10.19999981,   5.5999999 ,  -2.96802545,
         6.96000719, -10.4599905 ,  -3.22179294, -10.19824028,
         5.60174894,  -2.97676086,   6.96663904, -10.45340729,
         4.8402071 ,   1.58020127,   3.14647174], dtype=float32)

    Example of training neural network by conjugate gradient method
>>> tr.reset( )
>>> m = ConjugateGradient( ocl, n = 0.5, alpha = 0.3 )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
1.4140973
>>> tr.optimal_weights
array([ -3.22067547, -10.19999981,   5.5999999 ,  -2.96802545,
         6.96000719, -10.4599905 ,  -3.22179294, -10.19824028,
         5.60174894,  -2.97676086,   6.96663904, -10.45340729,
         4.8402071 ,   1.58020127,   3.14647174], dtype=float32)

    Example of training neural network by Quickprop method
>>> tr.reset( )
>>> m = Quickprop( ocl, n = 0.5, alpha = 0.3 )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
1.4141175
>>> tr.optimal_weights
array([ -3.22005892, -10.19999981,   5.5999999 ,  -2.96982789,
         6.96000195, -10.45999813,  -3.22009492, -10.19991112,
         5.60008574,  -2.97036433,   6.96034002, -10.45966911,
         4.83914328,   1.57822061,   3.15191698], dtype=float32)
         
    Example of training neural network by RPROP method
>>> tr.reset( )
>>> m = RPROP( ocl, n = 0.5 )
>>> i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> h.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
>>> o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
>>> m.start_training( nnc, training_data, tr, 10 )
>>> tr.minimal_error
0.012369672
>>> tr.optimal_weights
array([ -0.72000027,  -9.69999981,   6.0999999 ,  -3.66999984,
         6.96000004, -10.96000004,  -6.4800005 ,  -6.93999863,
        13.94000244,  -8.05000114,  12.04000187,  -5.37999821,
         5.66673517,   1.92594624,   3.74237275], dtype=float32)
"""

import numpy
import pyopencl
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
    
    Can be pickled.
    """

    def __init__( self, n = 0.5, alpha = 0.2, kw = 1.03, pd = 0.7, pi = 1.02 ):
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
        """
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

        pyopencl.enqueue_write_buffer( context.opencl.queue, context.weights_buf, weights, is_blocking = True )

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

        self.weights_delta_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
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

        start_time = time.clock()

        self.prepare_training( context )

        outputs_buf = pyopencl.Buffer( context.opencl.context, pyopencl.mem_flags.READ_ONLY, context.output_layer.neuron_count * 4 )
        total_error_buf = pyopencl.Buffer( context.opencl.context, pyopencl.mem_flags.READ_WRITE, context.output_layer.neuron_count * 4 )

        total_error = numpy.array( [ 0.0 for x in range( context.output_layer.neuron_count ) ], numpy.float32 )

        zero_buf = pyopencl.Buffer( 
            context.opencl.context,
            pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = total_error )

        i = 0
        while training_results.minimal_error > target_error:
            if i >= maximal_iterations:
                break
            i += 1

            pyopencl.enqueue_copy_buffer( context.opencl.queue, zero_buf, total_error_buf )
            for inputs, outputs in training_data:
                context.input_layer.set_inputs( inputs, is_blocking = False )
                context.input_layer.process()

                pyopencl.enqueue_write_buffer( context.opencl.queue, outputs_buf, outputs, is_blocking = False )
                context.opencl.kernel_setup_training_data( 
                    context.opencl.queue, ( context.output_layer.neuron_count, ),
                    context.outputs_buf,
                    numpy.int32( context.output_layer.neurons_offset ),
                    outputs_buf,
                    context.errors_backpropagation_buf,
                    total_error_buf
                    )

#                print context.output_layer.get_outputs()

                context.input_layer.calc_weights_gradient()
                self.adjust_weights( context )

            pyopencl.enqueue_read_buffer( context.opencl.queue, total_error_buf, total_error, is_blocking = True )

            error_sum = numpy.sqrt( total_error.sum() / len( training_data ) )

            self.adjust_training_parameters( error_sum )

            if error_sum < training_results.minimal_error:
                training_results.minimal_error = error_sum
                training_results.store_weights( context )

            if error_sum < target_error:
                break;

            training_results.opencl_time += context.opencl.gather_opencl_time()

        training_results.iterations += i
        training_results.total_time += time.clock() - start_time

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        dir = self.get_weights_direction_buf( context ) #this call should always return opposite direction

        context.opencl.kernel_adjust_weights( 
            context.opencl.queue, ( context.total_weights, ),
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

    def prepare_training( self, context ):
        """
        Create additional buffers to store previous gradient vector.
        
        @param layer
            Input layer.
        """
        super( ConjugateGradient, self ).prepare_training( context )

        self.direction_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )
        self.prev_gradient_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.01 for x in range( context.total_weights ) ], numpy.float32 )
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
            context.gradient_buf,
            self.prev_gradient_buf,
            numpy.int32( context.total_weights ),
            pyopencl.LocalMemory( 256 ),
            pyopencl.LocalMemory( 256 ),
            self.beta_buf,
            local_size = ( 64, )
            )

        self.iteration_count += 1
        if self.iteration_count > context.total_neurons:
            pyopencl.enqueue_write_buffer( context.opencl.queue, self.beta_buf, numpy.zeros( [1], numpy.float32 ), is_blocking = True )
            self.iteration_count = 0

#        test = numpy.ndarray( [ context.total_weights ], numpy.float32 )
#        pyopencl.enqueue_read_buffer( context.opencl.queue, self.beta_buf, test, is_blocking = True )

        context.opencl.kernel_calc_conjugate_gradient_direction( 
            context.opencl.queue, ( context.total_weights, ),
            context.gradient_buf,
            self.beta_buf,
            self.direction_buf,
            self.prev_gradient_buf
            )

#        pyopencl.enqueue_read_buffer( context.opencl.queue, self.direction_buf, test, is_blocking = True )

        return self.direction_buf



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
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )

    def __getstate__( self ):
        odict = super( ConjugateGradient, self ).__getstate__()
        del odict['prev_direction_buf']
        return odict

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        context.opencl.kernel_adjust_weights_quickprop( 
            context.opencl.queue, ( context.total_weights, ),
            context.gradient_buf,
            self.prev_direction_buf,
            self.n, self.alpha,
            self.weights_delta_buf,
            context.weights_buf
            )

        pyopencl.enqueue_copy_buffer( context.opencl.queue, context.gradient_buf, self.prev_direction_buf )

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
            hostbuf = numpy.array( [ self.n for x in range( context.total_weights ) ], numpy.float32 )
            )
        self.prev_gradient_buf = pyopencl.Buffer( 
            context.opencl.context, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( context.total_weights ) ], numpy.float32 )
            )

    def __getstate__( self ):
        odict = super( ConjugateGradient, self ).__getstate__()
        del odict['n_buf']
        del odict['prev_gradient_buf']
        return odict

    def adjust_weights( self, context ):
        """
        Adjust weights of neural network by certain direction.
        """
        context.opencl.kernel_adjust_weights_rprop( 
            context.opencl.queue, ( context.total_weights, ),
            context.gradient_buf,
            self.prev_gradient_buf,
            self.n_buf,
            context.weights_buf
            )

        #nn = numpy.ndarray( [context.total_weights], numpy.float32 )
        #pyopencl.enqueue_read_buffer( context.opencl.queue, context.gradient_buf, nn, is_blocking = True )
        #pyopencl.enqueue_read_buffer( context.opencl.queue, self.n_buf, nn, is_blocking = True )

        pyopencl.enqueue_copy_buffer( context.opencl.queue, context.gradient_buf, self.prev_gradient_buf )

    def adjust_training_parameters( self, error ):
        """
        Disable parameters adjustment since they interpreted differently.
        """
        pass



if __name__ == '__main__':
    import doctest
    doctest.testmod()
