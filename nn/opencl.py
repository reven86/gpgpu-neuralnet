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
                int inputs_ofs,
                int weights_ofs,
                int outputs_ofs,
                int inputs_per_neuron,
                int neuron_count,
                __local float * partial_sum,
                __global float * outputs )
            {
                for (uint y = get_group_id(0); y < neuron_count; y += get_num_groups(0))
                {
                    const __global float * row = weights + weights_ofs + y * inputs_per_neuron;

                    int lid = get_local_id(0);
                    int sum_start = lid;
                    
                    float sum;
                    if (lid == 0)
                    {
                        sum = row[0];
                        sum_start = get_local_size(0);
                    }
                    else
                        sum = 0.0f;

                    for (uint x = sum_start; x < inputs_per_neuron; x += get_local_size(0))
                        sum += row[x] * inputs[x - 1 + inputs_ofs];

                    partial_sum[lid] = sum;

                    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
                    {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        if (lid < stride)
                            partial_sum[lid] += partial_sum[lid + stride];
                    }

                    if (lid == 0)
                        outputs[y + outputs_ofs] = tanh( beta * partial_sum[0] );

                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            
            __kernel void calc_layer_gradient(
                __global const float * inputs,
                __global const float * errors,
                int inputs_ofs,
                int errors_ofs,
                int inputs_per_neuron,
                int grad_ofs,
                __global float * gradient )
            {
                int gid = get_global_id( 0 );
                
                int input_index = gid % inputs_per_neuron;
                int error_index = gid / inputs_per_neuron;
                
                if( input_index == 0 )
                    gradient[ gid + grad_ofs ] = errors[ errors_ofs + error_index ];
                else
                    gradient[ gid + grad_ofs ] = inputs[ inputs_ofs + input_index - 1 ] * errors[ errors_ofs + error_index ];
            }
            
            __kernel void propagate_errors(
                __global const float * errors,
                __global const float * weights,
                int errors_ofs,
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
                        sum += wcol[x * inputs_per_neuron] * errors[x + errors_ofs];

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
                int real_outputs_ofs,
                __global const float * target_outputs,
                __global float * errors,
                __global float * total_errors
                )
            {
                int gid = get_global_id( 0 );
                
                float err = real_outputs[ gid + real_outputs_ofs ] - target_outputs[ gid ];
                total_errors[ gid ] += err * err;
                
                errors[ gid + real_outputs_ofs ] = err * beta * ( 1.0f - real_outputs[ gid + real_outputs_ofs ] * real_outputs[ gid + real_outputs_ofs ] );
            }

            __kernel void adjust_weights(
                __global const float * direction,
                float n, float alpha,
                __global float * old_delta,
                __global float * weights
                )
            {
                int gid = get_global_id( 0 );
                
                float new_delta = n * ( -direction[ gid ] ) + alpha * old_delta[ gid ];
                
                weights[ gid ] += new_delta;
                old_delta[ gid ] = new_delta;
            }
            
            __kernel void adjust_weights_quickprop(
                __global const float * direction,
                __global const float * prev_direction,
                float n, float alpha, float gamma,
                __global float * old_delta,
                __global float * weights
                )
            {
                int gid = get_global_id( 0 );
                
                float new_delta;
                
                if( fabs( old_delta[ gid ] ) > 1e-8 )
                    new_delta = min( direction[ gid ] / ( prev_direction[ gid ] - direction[ gid ] ), alpha ) * old_delta[ gid ];
                else
                    new_delta = -n * direction[ gid ];
                
                weights[ gid ] += new_delta;
                old_delta[ gid ] = new_delta;
            }
            
            __kernel void calc_conjugate_gradient_beta(
                __global const float * gradient,
                __global const float * prev_gradient,
                int num_components,
                __local float * partial_sum_up,
                __local float * partial_sum_down,
                __global float * beta
                )
            {
                int lid = get_local_id( 0 );
            
                float up = 0.0f;
                float down = 0.0f;
                for (uint x = lid; x < num_components; x += get_local_size(0))
                {
                    up += gradient[ x ] * ( gradient[ x ] - prev_gradient[ x ] );
                    down += prev_gradient[ x ] * prev_gradient[ x ];
                }

                partial_sum_up[lid] = up;
                partial_sum_down[lid] = down;

                for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
                {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (lid < stride)
                    {
                        partial_sum_up[lid] += partial_sum_up[lid + stride];
                        partial_sum_down[lid] += partial_sum_down[lid + stride];
                    }
                }
                
                if( get_global_id( 0 ) == 0 )
                    beta[ 0 ] = fabs( partial_sum_down[ 0 ] ) < 1e-3 ? 0.0 : max( 0.0f, partial_sum_up[ 0 ] / partial_sum_down[ 0 ] );
            }
            
            __kernel void calc_conjugate_gradient_direction(
                __global const float * layer_gradient,
                __global const float * beta,
                __global float * direction,
                __global float * prev_gradient
                )
            {
                int gid = get_global_id( 0 );
                
                prev_gradient[ gid ] = layer_gradient[ gid ];               
                direction[ gid ] = layer_gradient[ gid ] + beta[ 0 ] * direction[ gid ];
            }
            """ ).build()

        self.kernel_process_layer = self.program.process_layer
        self.kernel_calc_layer_gradient = self.program.calc_layer_gradient
        self.kernel_propagate_errors = self.program.propagate_errors
        self.kernel_setup_training_data = self.program.setup_training_data
        self.kernel_adjust_weights = self.program.adjust_weights
        self.kernel_adjust_weights_quickprop = self.program.adjust_weights_quickprop
        self.kernel_calc_conjugate_gradient_beta = self.program.calc_conjugate_gradient_beta
        self.kernel_calc_conjugate_gradient_direction = self.program.calc_conjugate_gradient_direction
#        for attr in self.program.all_kernels():
#            setattr( self, 'kernel_' + attr.get_info( pyopencl.kernel_info.FUNCTION_NAME ), attr )

if __name__ == '__main__':
    import doctest
    doctest.testmod( optionflags = doctest.ELLIPSIS )
