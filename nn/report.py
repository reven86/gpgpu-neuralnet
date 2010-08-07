"""
Created on 01.06.2010

@author: RevEn
"""

import time
import csv

class Report( object ):
    """
    Report class. Gathers training information and creates different reports.
    """

    report_item = ( 'iteration', 'time', 'last_error', 'minimal_error',
                    'learning_rate', 'total_neurons', 'total_weights' )

    def __init__( self, filename ):
        """
        Report constructor.
        
        @param filename
            File name, where temporary stats will be stored.
        """

        self.entries = []
        self.filename = filename
        self.iteration = 0

        # erase file
        with open( filename, 'wb' ) as f:
            writer = csv.writer( f, delimiter = ";" )
            writer.writerow( Report.report_item )

    def __del__( self ):
        self.flush_stats()

    def process_iteration( self, iteration_delta, method, training_results, err, nnc ):
        """
        Gathers information for one training iteration.
        """
        self.iteration += iteration_delta

        self.entries.append( {
            'iteration' : self.iteration,
            'time' : time.clock(),
            'last_error' : err,
            'minimal_error' : training_results.minimal_error,
            'learning_rate' : method.n,
            'total_neurons' : nnc.total_neurons,
            'total_weights' : nnc.total_weights,
            } )

        if len( self.entries ) > 100:
            self.flush_stats()

    def flush_stats( self ):
        """
        Flush data to disk to free some memory.
        """

        with open( self.filename, 'ab' ) as f:
            writer = csv.writer( f, delimiter = ";" )
            for entry in self.entries:
                writer.writerow( [ entry[ x ] for x in Report.report_item ] )
            del self.entries[:]
