"""
Created on 14.05.2010

@author: RevEn

OpenCL programs and related data.
"""

import pyopencl #@UnresolvedImport
import numpy
import csv #@UnresolvedImport
import os

class OpenCL( object ):
    """
    This class stores all necessary data to perform parallel
    computations using OpenCL.
    """

    def __init__( self, opencl_context, enable_profiling = False ):
        """
        Constructs OpenCL queue, compiles all programs.

        @param opencl_context
            OpenCL context which will be used to execute computations.
            
        Example:
        
        >>> ocl = OpenCL( pyopencl.create_some_context( ), enable_profiling = True )
        >>> ocl.queue
        <pyopencl._cl.CommandQueue object at ...>
        """

        queue_fl = 0
        if enable_profiling:
            queue_fl = pyopencl.command_queue_properties.PROFILING_ENABLE

        self.context = opencl_context
        self.queue = pyopencl.CommandQueue( opencl_context, properties = queue_fl )
        self._profiling_enabled = enable_profiling
        self._max_local_size = map( int, self.context.devices[0].max_work_item_sizes )

        with open( os.path.join( os.path.dirname( __file__ ), 'ocl_programs.cl' ), 'rt' ) as f:
            self._program = pyopencl.Program( opencl_context, f.read() ).build()

        for attr in self._program.all_kernels():
            setattr( self, 'kernel_' + attr.get_info( pyopencl.kernel_info.FUNCTION_NAME ), attr )

        if self.profiling_enabled:
            ocl = self

            def profile_decorator( cmd ):
                class cmd2( object ):
                    def __call__( self, *kargs, **kwargs ):
                        evt = cmd( *kargs, **kwargs )
                        ocl._event_list.append( ( cmd.__name__, evt ) )
                        return evt
                return cmd2()

            def profile_kernel_decorator( cmd ):
                class cmd2( object ):
                    def __call__( self, *kargs, **kwargs ):
                        evt = cmd( *kargs, **kwargs )
                        ocl._event_list.append( ( kargs[1].get_info( pyopencl.kernel_info.FUNCTION_NAME ), evt ) )
                        return evt
                return cmd2()

            pyopencl.enqueue_copy_buffer = profile_decorator( pyopencl.enqueue_copy_buffer )
            pyopencl.enqueue_read_buffer = profile_decorator( pyopencl.enqueue_read_buffer )
            pyopencl.enqueue_write_buffer = profile_decorator( pyopencl.enqueue_write_buffer )
            pyopencl.enqueue_nd_range_kernel = profile_kernel_decorator( pyopencl.enqueue_nd_range_kernel )

            self._event_list = []
            self._event_times_by_kernel = {}

    @property
    def profiling_enabled( self ):
        return self._profiling_enabled

    @property
    def max_local_size( self ):
        return self._max_local_size

    def gather_opencl_stats( self ):
        """
        Returns time in seconds spent by OpenCL since last call to gather_opencl_time.
        """
        if not self.profiling_enabled:
            return 0.0

        # make sure all events are finished
        self.queue.finish()
        res = numpy.float32( 0.0 )

        # for each event and kernel compute time for each event state
        for k, e in self._event_list:
            times = numpy.array( ( 
                1,
                # event.profile works too slow
                1e-9 * ( e.get_profiling_info( pyopencl.profiling_info.SUBMIT ) - e.get_profiling_info( pyopencl.profiling_info.QUEUED ) ),
                1e-9 * ( e.get_profiling_info( pyopencl.profiling_info.START ) - e.get_profiling_info( pyopencl.profiling_info.SUBMIT ) ),
                1e-9 * ( e.get_profiling_info( pyopencl.profiling_info.END ) - e.get_profiling_info( pyopencl.profiling_info.START ) ),
                ), numpy.float32 )
            if k in self._event_times_by_kernel:
                self._event_times_by_kernel[k] += times
            else:
                self._event_times_by_kernel[k] = times
            res += times[3]

        del self._event_list[:]
        return res

    def flush_stats( self, filename ):
        """
        Dumps gathered kernel statistics to file
        """
        if self.profiling_enabled:
            with open( filename, 'wb' ) as f:
                writer = csv.writer( f, delimiter = ";" )
                for k, e in self._event_times_by_kernel.iteritems():
                    writer.writerow( [k] + list( e ) )

if __name__ == '__main__':
    import doctest #@UnresolvedImport
    doctest.testmod( optionflags = doctest.ELLIPSIS )
