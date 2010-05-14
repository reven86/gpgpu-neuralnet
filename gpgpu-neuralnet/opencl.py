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
            __kernel void process_layer(
                __global const float * inputs,
                __global const float * weights,
                int inputs_per_neuron,
                __global float * outputs )
            {
                int gid = get_global_id( 0 );
                
                float sum = 0.0f;
                for( int i = 0; i < inputs_per_neuron; i++ )
                    sum += inputs[ i ] * weights[ gid * inputs_per_neuron + i ];
                    
                outputs[ gid ] = tanh( 0.66666f * sum );
            }
            """ ).build()

if __name__ == '__main__':
    import doctest
    doctest.testmod( optionflags = doctest.ELLIPSIS )
