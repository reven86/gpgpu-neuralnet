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

def test():
    ocl = OpenCL( pyopencl.create_some_context() )
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
    #m = GradientDescent( n = 0.8, alpha = 0.3 )
    #m = ConjugateGradient( n = 0.8, alpha = 0.3 )
    #m = Quickprop( n = 0.8, alpha = 0.3 )
    m = RPROP( n = 0.8 )
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

        for t in training_data:
            i.set_inputs( t[0] )
            i.process()
            print o.get_outputs()

def test2():
    ocl = OpenCL( pyopencl.create_some_context(), enable_profiling = True )

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
    m = GradientDescent( n = 0.5 )        # 15, 8, 6, 5
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
                if data_bucket[-2] > 1.01:
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
                if data_bucket[-2] < 0.99:
                    test_bucket[9] = 1.0
                training_data.append( ( ( numpy.array( data_bucket[:-5], numpy.float32 ) - 1.0 ) * 20, test_bucket.copy() ) )
                data_bucket = data_bucket[5:]
                real_data_i.writerow( training_data[-1][0] )
                real_data_o.writerow( test_bucket )

        prev_line = raw_line

    m.randomize_weights( nnc )

#    with open( 'nn_data_7_2.92699610853.pkl', 'rb' ) as f:
#        tr, m = cPickle.load( f )
#        tr.apply_weights( nnc )

    for it in range( 100 ):
        m.start_training( nnc, training_data, tr, 1 )
        print "Error: ", tr.minimal_error
        print "Weights: ", tr.optimal_weights
        print "Iterations: ", tr.iterations
        print "OpenCL time: ", tr.opencl_time
        print "Total time: ", tr.total_time

        with open( ''.join( ( 'nn_data_', str( tr.iterations ), '_', str( tr.minimal_error ), '.pkl' ) ), 'wb' ) as f:
            cPickle.dump( ( tr, m ), f, -1 )

        for ti, t in enumerate( training_data ):
            nn_h.set_inputs( t[0] )
            nn_h.process()
            out = nn_o.get_outputs()
            err = out - t[ 1 ];
            print out, numpy.sqrt( ( err * err ).sum() )
            if ti > 5:
                break;

if __name__ == '__main__':
    cProfile.run( 'test2( )', 'test_prof' )
