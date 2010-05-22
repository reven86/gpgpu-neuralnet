'''
Created on 19.05.2010

@author: RevEn
'''

from nn.opencl import *
from nn.layer import *
from nn.training import *
import cProfile

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
    #m = GradientDescent( ocl, n = 0.8, alpha = 0.3 )
    #m = ConjugateGradient( ocl, n = 0.8, alpha = 0.3 )
    m = Quickprop( ocl, n = 0.8, alpha = 0.3 )
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

    for it in range( 3 ):
        m.start_training( nnc, training_data, tr, 10000 )
        print "Error: ", tr.minimal_error
        print "Weights: ", tr.optimal_weights
        print "Iterations: ", tr.iterations

        for t in training_data:
            i.set_inputs( t[0] )
            i.process()
            print o.get_outputs()

if __name__ == '__main__':
    cProfile.run( 'test( )', 'test_prof' )
