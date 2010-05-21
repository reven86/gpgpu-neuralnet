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
    i.finilize_links()

    #i.set_weights( numpy.array( ( -3.22, -10.2, 5.6, -2.97, 6.96, -10.46 ), numpy.float32 ) )
    #o.set_weights( numpy.array( ( 4.839, 1.578, 3.152 ), numpy.float32 ) )
    i.set_weights( numpy.array( ( 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ), numpy.float32 ) )
    o.set_weights( numpy.array( ( 0.5, 0.5, 0.5 ), numpy.float32 ) )

    tr = TrainingResults()
    gd = GradientDescent( ocl, n = 0.8, alpha = 0.3 )
    training_data = ( 
        ( numpy.array( ( 0.0, 0.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
        ( numpy.array( ( 0.0, 1.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
        ( numpy.array( ( 1.0, 0.0, ), numpy.float32 ), numpy.array( ( 1.0, ), numpy.float32 ) ),
        ( numpy.array( ( 1.0, 1.0, ), numpy.float32 ), numpy.array( ( 0.0, ), numpy.float32 ) ),
        )

    #gd.randomize_weights( i )

    for it in range( 100 ):
        gd.start_training( i, o, training_data, tr, 10000 )
        print "Error: ", tr.minimal_error
        print "Weights: ", tr.optimal_weights
        print "Iterations: ", tr.iterations
        i.set_inputs( training_data[0][0] )
        i.process()
        print o.get_outputs()
        i.set_inputs( training_data[1][0] )
        i.process()
        print o.get_outputs()
        i.set_inputs( training_data[2][0] )
        i.process()
        print o.get_outputs()
        i.set_inputs( training_data[3][0] )
        i.process()
        print o.get_outputs()

if __name__ == '__main__':
    cProfile.run( 'test( )', 'test_prof' )
