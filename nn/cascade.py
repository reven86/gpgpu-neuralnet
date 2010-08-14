"""
Created on 07.08.2010

@author: RevEn

This module provides functionality to train and test
Cascade Correlation Neural Networks (Fahlman and Libiere, 1990)
"""

import pyopencl
import numpy

class CascadeCorrelationNeuralNetwork( object ):
    """
    Cascaded Correlation Neural Network.
    
    This class manages entire neural network, its creation, training,
    testing, saving and loading
    """


    def __init__( self, opencl, input_count, output_count ):
        """
        Creates neural network without hidden layers.
        
        @param opencl
            OpenCL context.
            
        @param input_count.
            Inputs count.
            
        @param output_count.
            Outputs count.
        """

        self.opencl = opencl
        self.total_inputs = input_count
        self.total_outputs = output_count

        self.total_neurons = input_count + output_count
        self.total_weights = ( input_count + 1 ) * output_count

        self.neurons_buf_size = 16 * ( 1 + self.total_neurons // 16 )
        self.weights_buf_size = 16 * ( 1 + self.total_weights // 16 )

        self.neurons_buf = pyopencl.Buffer( 
            self.opencl.context,
            pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [self.neurons_buf_size], numpy.float32 )
            )

        self.weights_buf = pyopencl.Buffer( 
            self.opencl.context,
            pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf = numpy.zeros( [self.weights_buf_size], numpy.float32 )
            )

    def start_training( self, training_data, training_results,
                        maximal_neurons = 300, candidates_count = 30,
                        target_error = 0.01, report = False ):
        """
        Starts training.
        
        @param training_data
            Array of tuples of inputs and outputs.
            
        @param training_results
            TrainingResults structure where optimal results will be stored.
            
        @param maximal_neurons
            Maximal additional errors that can be added during this training method invocation.
            
        @param candidates_count
            Neuron candidates count for each step.
            
        @param target_error
            Target absolute error.
            
        @param report
            Report object (optimal)
            
        @return Tuple of performed iterations count, minimal relative error
        """

        pass
