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
>>> i.finilize_links( )

    Example of usage TrainingResults to store and restore neural network weights
>>> tr = TrainingResults( )
>>> h.set_weights( numpy.array( [ 1, 1, 1, 1, 1, 1 ], dtype=numpy.float32 ) )
>>> tr.store_weights( i )
>>> old_weights = tr.optimal_weights
>>> h.get_weights( )
array([ 1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)
>>> h.set_weights( numpy.array( [ 2, 2, 2, 2, 2, 2 ], dtype=numpy.float32 ) )
>>> h.get_weights( )
array([ 2.,  2.,  2.,  2.,  2.,  2.], dtype=float32)
>>> tr.apply_weights( i )
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
>>> gd.randomize_weights( i )
>>> gd.start_training( i, o, training_data, tr, 100 )
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

    def store_weights( self, layer, start = True ):
        """
        Stores list of layers weights for entire neural network.
        """

        if start:
            self.optimal_weights = []

        self.optimal_weights.append( layer.get_weights() )
        for l in layer.next_layers:
            self.store_weights( l[0], False )

    def apply_weights( self, layer, i = 0 ):
        """
        Apply optimal weights to neural network.
        """

        layer.set_weights( self.optimal_weights[ i ] )
        for l in layer.next_layers:
            self.apply_weights( l[0], i + 1 )

class TrainingMethod( object ):
    """
    Base class for all neural network training methods.
    """

    def __init__( self, opencl ):
        """
        Construct base training method object.
        """
        self.opencl = opencl

    def randomize_weights( self, layer ):
        """
        Initialize weights of layer by random values
        """

        weights = numpy.random.rand( layer.inputs_per_neuron * layer.neuron_count ).astype( numpy.float32 )
        weights -= 0.5
        weights *= 4.0 / numpy.sqrt( numpy.float32( layer.inputs_per_neuron ) )

        layer.set_weights( weights )
        for l in layer.next_layers:
            self.randomize_weights( l[0] )

    def start_training( self, input_layer, output_layer, training_data, training_results,
                        maximal_iterations = 10000, target_error = 0.0001 ):
        """
        Starts training.
        
        @param input_layer
            Input layer of neural network.
            
        @param output_layer
            Output layer of neural network.
            
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

        self.prepare_training( input_layer )

        i = 0
        while training_results.minimal_error > target_error:
            i += 1
            if i > maximal_iterations:
                break

            error_sum = numpy.float32( 0.0 );
            for inputs, outputs in training_data:
                input_layer.set_inputs( inputs )
                input_layer.process()

                cur_error = output_layer.get_outputs() - outputs
                output_layer.setup_training_data( cur_error )

                cur_error *= cur_error
                cur_error = cur_error.sum()
                error_sum += cur_error

                input_layer.calc_weights_gradient()

                #print input_layer.next_layers[0][0].get_weights()
                self.adjust_weights( input_layer )
                #print input_layer.next_layers[0][0].get_weights()

            if error_sum < training_results.minimal_error:
                training_results.minimal_error = error_sum
                training_results.store_weights( input_layer )

            if error_sum < target_error:
                break;

        training_results.iterations += i

class GradientDescent( TrainingMethod ):
    """
    Gradient descent optimization method. Uses gradient as a vector of weights adjustment.
    """

    def __init__( self, opencl, n = 0.5, alpha = 0.5 ):
        """
        Constructs gradient descent training method.
        
        @param n
            Training coefficient [0, 1].
            
        @param alpha
            Momentum coefficient. Useful when the gradient is near to zero vector.
        """
        super( GradientDescent, self ).__init__( opencl )

        self.n = numpy.float32( n )
        self.alpha = numpy.float32( alpha )

    def prepare_training( self, layer ):
        """
        Initialize training method by neural network.
        
        @param layer
            Input layer.
        """

        ll = [ layer ]
        self.total_weights = numpy.int32( 0.0 )

        while ll:
            l = ll.pop()
            self.total_weights += l.inputs_per_neuron * l.neuron_count
            for k in l.next_layers:
                ll.append( k[0] )

        self.weights_delta_buf = pyopencl.Buffer( 
            self.opencl.context, pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.array( [ 0.0 for x in range( self.total_weights ) ], numpy.float32 )
            )
        self.weights_delta_offset = numpy.int32( 0 )

    def adjust_weights( self, layer ):
        """
        Adjust weights of neural network by applying gradient descent method.
        """
        #gradients = numpy.ndarray( [layer.neuron_count * layer.inputs_per_neuron], numpy.float32 )
        #pyopencl.enqueue_read_buffer( self.opencl.queue, layer.gradients_buf, gradients, is_blocking = True )

        self.opencl.program.adjust_weights_gradient_descent( 
            self.opencl.queue, ( layer.neuron_count * layer.inputs_per_neuron, ),
            layer.gradients_buf,
            self.n, self.alpha,
            self.weights_delta_offset,
            self.weights_delta_buf,
            layer.weights_buf
            )

        self.weights_delta_offset = ( self.weights_delta_offset + layer.inputs_per_neuron * layer.neuron_count ) % self.total_weights

        for l in layer.next_layers:
            self.adjust_weights( l[ 0 ] )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
