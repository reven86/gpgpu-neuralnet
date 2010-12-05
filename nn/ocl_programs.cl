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
    __global float * gradient,
    int weights_count,
    __local float * temp_mem )
{
	int i = get_global_id( 0 );

	__global const float * err = errors + errors_ofs;
	__global const float * inp = inputs + inputs_ofs;
	__global float * grad = gradient + grad_ofs;

    int input_index = i % inputs_per_neuron;
    int error_index = i / inputs_per_neuron;
    
    int lid = get_local_id( 0 );
    int lid_inp = lid % inputs_per_neuron;
    int lid_div_inp = lid / inputs_per_neuron;
    
    if( lid < inputs_per_neuron )
    	temp_mem[ lid ] = input_index == 0 ? 1.0 : inp[ input_index - 1 ];
    if( lid_inp == 0 )
    	temp_mem[ inputs_per_neuron + lid_div_inp ] = err[ error_index ];
    	
    float g = grad[ i ];
    
    barrier(CLK_LOCAL_MEM_FENCE);

	if( i >= weights_count )
		return;

    float e = temp_mem[ inputs_per_neuron + lid_div_inp ] * temp_mem[ lid_inp ];
	grad[ i ] = g + e;
}

__kernel void propagate_errors(
    __global float * errors,
    __global const float * weights,
    int errors_ofs,
    int ofs, int count,
    int errors_count,
    int weights_offset,
    int inputs_per_neuron,
    __local float * partial_sum,
    __global const float * outputs
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
            errors[ y + ofs ] += partial_sum[ 0 ] * beta * ( 1.0f - output * output );
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void setup_training_data(
    int total_count,
    __global const float * real_outputs,
    int real_outputs_ofs,
    int real_outputs_count,
    __global const float * target_outputs,
    __local float * partial_sum,
    __global float * errors,
    __global float * total_errors
    )
{
    int lid = get_local_id( 0 );

    float sum = 0.0f;
    for (uint x = lid; x < total_count; x += get_local_size(0))
    {
        if( x < real_outputs_ofs || x >= real_outputs_ofs + real_outputs_count )
        {
            errors[ x ] = 0.0f;
            continue;
        }
        
        float err = real_outputs[ x ] - target_outputs[ x - real_outputs_ofs ];
        errors[ x ] = err * beta * ( 1.0f - real_outputs[ x ] * real_outputs[ x ] );

        sum += err * err;
    }

    partial_sum[lid] = sum;

    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride)
            partial_sum[lid] += partial_sum[lid + stride];
    }
    
    if( get_global_id( 0 ) == 0 )
        total_errors[ 0 ] += native_sqrt( partial_sum[ 0 ] );
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
    float n, float alpha,
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

__kernel void adjust_weights_rprop(
    __global const float * gradient,
    __global const float * prev_gradient,
    __global float * n,
    __global float * weights
    )
{
    int gid = get_global_id( 0 );
    
    if( fabs( gradient[ gid ] ) < 1e-6f )
        return;
    
    float new_delta;
    float factor = prev_gradient[ gid ] * gradient[ gid ];
    
    const float a = 1.2f;
    const float b = 0.5f;
    
    if( factor > 1e-6f )
        n[ gid ] = min( a * n[ gid ], 50.0f );
    else if ( factor < -1e-6f )
        n[ gid ] = max( b * n[ gid ], 1e-6f );
    
    weights[ gid ] += n[ gid ] * ( gradient[ gid ] > 0.0f ? -1.0f : 1.0f );
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
