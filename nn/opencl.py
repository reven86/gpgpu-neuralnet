"""
Created on 14.05.2010

@author: RevEn

OpenCL programs and related data.
"""

import pyopencl

class OpenCL( object ):
    """
    This class stores all necessary data to perform parallel
    computations using OpenCL.
    """

    def __init__( self, opencl_context ):
        """
        Constructs OpenCL queue, compiles all programs.

        @param opencl_context
            OpenCL context which will be used to execute computations.
            
        Example:
        
        >>> ocl = OpenCL( pyopencl.create_some_context( ) )
        >>> ocl.program
        <pyopencl._cl.Program object at ...>
        >>> ocl.queue
        <pyopencl._cl.CommandQueue object at ...>
        """

        self.context = opencl_context
        self.queue = pyopencl.CommandQueue( opencl_context )
        self.program = pyopencl.Program( opencl_context, """
            __constant float beta = 0.66666f;

            __kernel void process_layer(
                __global const float * inputs,
                __global const float * weights,
                int inputs_per_neuron,
                __global float * outputs )
            {
                int gid = get_global_id( 0 );
                
                float sum = weights[ gid * inputs_per_neuron ];    // polarization link
                for( int i = 0; i < inputs_per_neuron - 1; i++ )
                    sum += inputs[ i ] * weights[ gid * inputs_per_neuron + i + 1 ];
               
                outputs[ gid ] = tanh( beta * sum );
            }
            
            __kernel void calc_derivatives(
                __global const float * outputs,
                __global float * errors )
            {
                int gid = get_global_id( 0 );
                
                float derivative = beta * ( 1.0f - outputs[ gid ] * outputs[ gid ] );

                errors[ gid ] = derivative * errors[ gid ];
            }
            
            __kernel void calc_layer_gradients(
                __global const float * inputs,
                __global const float * errors,
                int inputs_per_neuron,
                __global float * gradients )
            {
                int gid = get_global_id( 0 );
                
                int input_index = gid % inputs_per_neuron;
                int error_index = gid / inputs_per_neuron;
                
                float input;
                if( input_index == 0 )
                    input = 1.0f;
                else
                    input = inputs[ input_index - 1 ];

                gradients[ gid ] = input * errors[ error_index ];
            }
            
            __kernel void propagate_errors(
                __global const float * errors,
                __global const float * weights,
                int ofs,
                int errors_count,
                int weights_offset,
                int inputs_per_neuron,
                __global float * new_errors
                )
            {
                int gid = get_global_id( 0 );
                
                float sum = 0.0f;
                for( int i = 0; i < errors_count; i++ )
                    sum += errors[ i ] * weights[ weights_offset + i * inputs_per_neuron ];
                
                new_errors[ gid + ofs ] = sum;
            }
            
            __kernel void adjust_weights_gradient_descent(
                __global const float * gradients,
                float n, float alpha,
                int delta_offset,
                __global float * old_delta,
                __global float * weights
                )
            {
                int gid = get_global_id( 0 );
                
                float new_delta = n * ( -gradients[ gid ] ) + alpha * old_delta[ gid + delta_offset ];
                
                weights[ gid ] += new_delta;
                old_delta[ gid + delta_offset ] = new_delta;
            }
            """ ).build()

if __name__ == '__main__':
    import doctest
    doctest.testmod( optionflags = doctest.ELLIPSIS )
