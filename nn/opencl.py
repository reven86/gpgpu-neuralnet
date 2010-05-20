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
                int neuron_count,
                __local float * partial_sum,
                __global float * outputs )
            {
                for (uint y = get_group_id(0); y < neuron_count; y += get_num_groups(0))
                {
                    const __global float * row = weights + y * inputs_per_neuron;

                    int lid = get_local_id(0);
                    int sum_start = lid;
                    
                    float sum;
                    if (lid == 0)
                    {
                        sum = row[0];
                        sum_start = get_local_size(0);
                    }
                    else
                        sum = 0.0;

                    for (uint x = sum_start; x < inputs_per_neuron; x += get_local_size(0))
                        sum += row[x] * inputs[x - 1];

                    partial_sum[lid] = sum;

                    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
                    {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        if (lid < stride)
                            partial_sum[lid] += partial_sum[lid + stride];
                    }

                    if (lid == 0)
                        outputs[y] = tanh( beta * partial_sum[0] );

                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            
            __kernel void calc_layer_gradient(
                __global const float * inputs,
                __global const float * errors,
                int inputs_per_neuron,
                __global float * gradient )
            {
                int gid = get_global_id( 0 );
                
                int input_index = gid % inputs_per_neuron;
                int error_index = gid / inputs_per_neuron;
                
                if( input_index == 0 )
                    gradient[ gid ] = errors[ error_index ];
                else
                    gradient[ gid ] = inputs[ input_index - 1 ] * errors[ error_index ];
            }
            
            __kernel void propagate_errors(
                __global const float * errors,
                __global const float * weights,
                int ofs, int count,
                int errors_count,
                int weights_offset,
                int inputs_per_neuron,
                __local float * partial_sum,
                __global const float * outputs,
                __global float * new_errors
                )
            {
                for (uint y = get_group_id(0); y < count; y += get_num_groups(0))
                {
                    const __global float * wcol = weights + weights_offset + y;

                    int lid = get_local_id(0);                    

                    float sum = 0.0f;
                    for (uint x = lid; x < errors_count; x += get_local_size(0))
                        sum += wcol[x * inputs_per_neuron] * errors[x];

                    partial_sum[lid] = sum;

                    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
                    {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        if (lid < stride)
                            partial_sum[lid] += partial_sum[lid + stride];
                    }

                    if (lid == 0)
                    {
                        float output = outputs[ y + ofs ];
                        new_errors[ y + ofs ] = partial_sum[ 0 ] * beta * ( 1.0f - output * output );
                    }

                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            
            __kernel void setup_training_data(
                __global const float * real_outputs,
                __global const float * target_outputs,
                __global float * errors,
                __global float * total_errors
                )
            {
                int gid = get_global_id( 0 );
                
                float err = real_outputs[ gid ] - target_outputs[ gid ];
                total_errors[ gid ] += err * err;
                
                errors[ gid ] = err * beta * ( 1.0f - real_outputs[ gid ] * real_outputs[ gid ] );
            }

            __kernel void adjust_weights_gradient_descent(
                __global const float * gradient,
                float n, float alpha,
                int delta_offset,
                __global float * old_delta,
                __global float * weights
                )
            {
                int gid = get_global_id( 0 );
                
                float new_delta = n * ( -gradient[ gid ] ) + alpha * old_delta[ gid + delta_offset ];
                
                weights[ gid ] += new_delta;
                old_delta[ gid + delta_offset ] = new_delta;
            }
            """ ).build()

        self.kernel_process_layer = self.program.process_layer
        self.kernel_calc_layer_gradient = self.program.calc_layer_gradient
        self.kernel_propagate_errors = self.program.propagate_errors
        self.kernel_setup_training_data = self.program.setup_training_data
        self.kernel_adjust_weights_gradient_descent = self.program.adjust_weights_gradient_descent
#        for attr in self.program.all_kernels():
#            setattr( self, 'kernel_' + attr.get_info( pyopencl.kernel_info.FUNCTION_NAME ), attr )

if __name__ == '__main__':
    import doctest
    doctest.testmod( optionflags = doctest.ELLIPSIS )
