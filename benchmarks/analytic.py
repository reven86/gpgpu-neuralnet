'''
Created on 04.07.2010

@author: RevEn
'''

import numpy
from nn.opencl import *
from nn.layer import *
from nn.training import *
from math import sin, cos

if __name__ == '__main__':

    funcs = ( 
        lambda x: 0.5,
        lambda x: 0.5 + 0.125 * sin( 20.0 * x ) + 0.25 * cos( 10.0 * x ),
        lambda x: 0.5 + 0.5 * sin( 30.0 * x ) / ( 1.0 + x * x )
        )

    ocl = OpenCL( pyopencl.create_some_context(), enable_profiling = True )
    i = InputLayer( 5, ocl )
    h = Layer( 10, ocl )
    o = OutputLayer( 1, ocl )
    i.link_next( h )
    h.link_next( o )
    nnc = ExecutionContext( i, o, allow_training = True )

    for f in funcs:
        data = numpy.array( [ f( x / 40.0 ) for x in xrange( 40 ) ], numpy.float32 )

        tr = TrainingResults()
        m = GradientDescent()
        m.randomize_weights( nnc )

        training_data = []
        for x in range( 0, len( data ) - 5 ):
            training_data.append( ( data[x:x + 5], numpy.array( ( data[x + 5], ), numpy.float32 ) ) )

        m.start_training( nnc, training_data[:15], tr, 10000, 1e-12 )

        print f
        for ti, t in enumerate( training_data ):
            i.set_inputs( t[0] )
            i.process()
            print "{0},{1},{2}".format( ti, t[1][0], o.get_outputs()[0] )
