'''
Created on 19.05.2010

@author: RevEn
'''

from nn.opencl import *
from nn.layer import *
from nn.training import *
import cProfile
import cPickle
import csv
import random

def test():
    ocl = OpenCL( pyopencl.create_some_context(), enable_profiling = True )
    i = InputLayer( 2, ocl )
    h1 = Layer( 1000, ocl )
    h2 = Layer( 10, ocl )
    o = OutputLayer( 1, ocl )
    i.link_next( o, 0, 2 )
    #h1.link_next( o, 0, 1000 )
    #h2.link_next( o, 0, 10 )
    nnc = ExecutionContext( i, o, allow_training = True )

    i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
    o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
    #i.set_weights( numpy.array( ( 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ), numpy.float32 ) )
    #o.set_weights( numpy.array( ( 0.5, 0.5, 0.5 ), numpy.float32 ) )

    tr = TrainingResults()
    m = GradientDescent()
    #m = ConjugateGradient( n = 0.8, alpha = 0.3 )
    #m = Quickprop( n = 0.8, alpha = 0.3 )
    #m = RPROP( n = 0.8 )
    training_data = ( 
        ( numpy.array( ( 0.0, 0.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
        ( numpy.array( ( 0.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
        ( numpy.array( ( 1.0, 0.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
        ( numpy.array( ( 1.0, 1.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
        )

    #m.randomize_weights( nnc )

    # GradientDescent - 10007 iterations to convergence (~40s GeForce 8800 GT)
    # ConjugateGradient - >30000 iterations to convergence (bug??) (~130s GeForce 8800 GT)
    # QuickProp - >30000 iterations to convergence
    # RPROP - >30000 iterations to convergence

    for it in range( 3 ):
        m.start_training( nnc, training_data, tr, 10000 )
        print "Error: ", tr.minimal_error
        print "Weights: ", tr.optimal_weights
        print "Iterations: ", tr.iterations
        print "OpenCL time: ", tr.opencl_time
        print "Total time: ", tr.total_time

        for t in training_data:
            i.set_inputs( t[0] )
            i.process()
            print o.get_outputs()

def test2():
    ocl = OpenCL( pyopencl.create_some_context(), enable_profiling = True )
    print ocl.context.devices[0].type

    h_count = 600
    d_count = 5
    i_count = 2

    nn_h = InputLayer( h_count, ocl )
    nn_d = [ Layer( 5, ocl ) for i in range( d_count ) ]
    nn_w = Layer( 5, ocl )
    nn_i = [ Layer( 5, ocl ) for i in range( i_count ) ]
    nn_o = OutputLayer( 10, ocl )

    for li, l in enumerate( nn_d ):
        nn_h.link_next( l, li * h_count / d_count, h_count / d_count )
        l.link_next( nn_w, 0, 5 )
        l.link_next( nn_o, 0, 5 )

    for li, l in enumerate( nn_i ):
        nn_h.link_next( l, h_count - 20 * ( li + 1 ), 20 * ( li + 1 ) )
        l.link_next( nn_o, 0, 5 )

    nn_w.link_next( nn_o, 0, 5 )
    nn_h.link_next( nn_o, h_count - 15, 15 )

    nnc = ExecutionContext( nn_h, nn_o, allow_training = True )

    tr = TrainingResults()

    m = GradientDescent()        # 15, 8, 6, 5
    #m = ConjugateGradient( )    # 31, 31
    #m = Quickprop( )        # 28
    #m = RPROP( )            # 24, 24, 24

    training_data = []
    raw_data = csv.reader( open( 'raw.csv', 'rt' ) )
    real_data_i = csv.writer( open( 'training_i.csv', 'wb' ) )
    real_data_o = csv.writer( open( 'training_o.csv', 'wb' ) )
    prev_line = None
    data_bucket = []
    test_bucket = numpy.zeros( [nn_o.neuron_count], numpy.float32 )
    for raw_line in raw_data:
        if prev_line:
            data_bucket.extend( numpy.array( map( float, raw_line[-5:] ), numpy.float32 ) / numpy.array( map( float, prev_line[-5:] ), numpy.float32 ) )
            if len( data_bucket ) > nn_h.neuron_count:
                test_bucket.fill( 0.0 )
                if data_bucket[-2] > 1.007:
                    test_bucket[0] = 1.0
                if data_bucket[-2] > 1.005:
                    test_bucket[1] = 1.0
                if data_bucket[-2] > 1.003:
                    test_bucket[2] = 1.0
                if data_bucket[-2] > 1.002:
                    test_bucket[3] = 1.0
                if data_bucket[-2] > 1.001:
                    test_bucket[4] = 1.0
                if data_bucket[-2] < 0.999:
                    test_bucket[5] = 1.0
                if data_bucket[-2] < 0.998:
                    test_bucket[6] = 1.0
                if data_bucket[-2] < 0.997:
                    test_bucket[7] = 1.0
                if data_bucket[-2] < 0.995:
                    test_bucket[8] = 1.0
                if data_bucket[-2] < 0.993:
                    test_bucket[9] = 1.0
                training_data.append( ( ( numpy.array( data_bucket[:-5], numpy.float32 ) - 1.0 ) * 20, test_bucket.copy() ) )
                data_bucket = data_bucket[5:]
                real_data_i.writerow( training_data[-1][0] )
                real_data_o.writerow( test_bucket )

        prev_line = raw_line

    m.randomize_weights( nnc )
    #del training_data[:200]

    q_start = 0
    q_size = 3 * 22 * 24

    with open( 'nn_data_158389_559_0.0603293361086.pkl', 'rb' ) as f:
        tr, m, q_start = cPickle.load( f )
        tr.apply_weights( nnc )

    tr.minimal_error = 1e12
    tr.total_time = 0.0
    tr.opencl_time = 0.0
    q_start = len( training_data ) - q_size;

    training_data1q = training_data[q_start:q_start + q_size]

    target_error = 0.01

    for it in range( 10000 ):
        m.start_training( nnc, training_data1q, tr, 100, target_error )
        print "Error: ", tr.minimal_error
        print "Weights: ", tr.optimal_weights
        print "Iterations: ", tr.iterations
        print "OpenCL time: ", tr.opencl_time
        print "Total time: ", tr.total_time
        print "N: ", m.n
        print "Data start: ", q_start

        with open( ''.join( ( 'nn_data_', str( tr.iterations ), '_', str( q_start ), '_', str( tr.minimal_error ), '.pkl' ) ), 'wb' ) as f:
            cPickle.dump( ( tr, m, q_start ), f, -1 )

        if tr.minimal_error < target_error:
            q_start += 1
            training_data1q = training_data[q_start:q_start + q_size]
            if len( training_data1q ) < q_size:
                break;
            tr.minimal_error = 1e12

        for ti, t in enumerate( training_data ):
            nn_h.set_inputs( t[0] )
            nn_h.process()
            out = nn_o.get_outputs()
            err = out - t[ 1 ];
            print out, numpy.sqrt( ( err * err ).sum() )
            if ti > 5:
                break;

def test3():
    ocl = OpenCL( pyopencl.create_some_context(), enable_profiling = True )
    print ocl.context.devices[0].type

    m15_count = 24 * 4 * 2

    nn_m15 = InputLayer( m15_count, ocl )
    nn_h1 = Layer( m15_count * 4, ocl )
    nn_h2 = Layer( m15_count, ocl )
    nn_o = OutputLayer( 1, ocl )

    nn_m15.link_next( nn_h1 )
    nn_h1.link_next( nn_h2 )
    nn_h2.link_next( nn_o )

    nnc = ExecutionContext( nn_m15, nn_o, allow_training = True )

    tr = TrainingResults()

    m = GradientDescent()        # 15, 8, 6, 5
    #m = ConjugateGradient( )    # 31, 31
    #m = Quickprop( )        # 28
    #m = RPROP( )            # 24, 24, 24

    dates = []
    training_data = []
    raw_data = csv.reader( open( 'raw_m15.csv', 'rt' ) )
    real_data_i = csv.writer( open( 'training_i_m15.csv', 'wb' ) )
    real_data_o = csv.writer( open( 'training_o_m15.csv', 'wb' ) )
    prev_line = None
    data_bucket = []
    test_bucket = numpy.zeros( [nn_o.neuron_count], numpy.float32 )
    for raw_line in raw_data:
        if prev_line:
            data_bucket.extend( numpy.array( map( float, raw_line[-2:] ), numpy.float32 ) / numpy.array( map( float, prev_line[-2:] ), numpy.float32 ) )
            if len( data_bucket ) > nn_m15.neuron_count:
                test_bucket.fill( 0.0 )
                if data_bucket[-2] > 1.001:
                    test_bucket[0] = 1.0
                if data_bucket[-2] < 0.999:
                    test_bucket[0] = -1.0
                training_data.append( ( ( numpy.array( data_bucket[:-2], numpy.float32 ) - 1.0 ) * 20, test_bucket.copy() ) )
                data_bucket = data_bucket[2:]
                real_data_i.writerow( training_data[-1][0] )
                real_data_o.writerow( test_bucket )
                dates.append( raw_line[ 0:2 ] )

        prev_line = raw_line

    m.randomize_weights( nnc )
    #del training_data[:200]

    q_start = 0
    q_size = 3 * 22 * 24 * 4

    with open( 'nn_data_m15_26106_1213_0.00995071548404.pkl', 'rb' ) as f:
        tr, m, q_start = cPickle.load( f )
        tr.apply_weights( nnc )

    tr.minimal_error = 1e12
    tr.total_time = 0.0
    tr.opencl_time = 0.0
    #q_start = len( training_data ) - q_size;

    for i, t in enumerate( training_data ):
        nnc.input_layer.set_inputs( t[0] )
        nnc.input_layer.process()
        out = nnc.output_layer.get_outputs()
        err = out - t[1]
        print i, dates[i], numpy.sqrt( err * err ).sum(), list( t[1] ), list( out )
        #print out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9]

#    training_data1q = training_data[q_start:q_start + q_size]
#
#    target_error = 0.01
#
#    for it in range( 10000 ):
#        m.start_training( nnc, training_data1q, tr, 100, target_error )
#        print "Error: ", tr.minimal_error
#        print "Weights: ", tr.optimal_weights
#        print "Iterations: ", tr.iterations
#        print "OpenCL time: ", tr.opencl_time
#        print "Total time: ", tr.total_time
#        print "N: ", m.n
#        print "Data start: ", q_start
#
#        with open( ''.join( ( 'nn_data_m15_', str( tr.iterations ), '_', str( q_start ), '_', str( tr.minimal_error ), '.pkl' ) ), 'wb' ) as f:
#            cPickle.dump( ( tr, m, q_start ), f, -1 )
#
#        if tr.minimal_error < target_error:
#            q_start += 1
#            training_data1q = training_data[q_start:q_start + q_size]
#            if len( training_data1q ) < q_size:
#                break;
#            tr.minimal_error = 1e12
#
#        for ti, t in enumerate( training_data ):
#            nn_m15.set_inputs( t[0] )
#            nn_m15.process()
#            out = nn_o.get_outputs()
#            err = out - t[ 1 ];
#            print out, numpy.sqrt( ( err * err ).sum() )
#            if ti > 5:
#                break;

if __name__ == '__main__':
    cProfile.run( 'test3( )', 'test_prof' )
    #test2( )
