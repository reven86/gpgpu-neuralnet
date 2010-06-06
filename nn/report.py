"""
Created on 01.06.2010

@author: RevEn
"""

import time

class Report( object ):
    """
    Report class. Gathers training information and creates different reports.
    """

    def __init__( self ):
        """
        Empty constructor.
        """

        self.entries = []

    def process_iteration( self, method, training_results, err, nnc ):
        """
        Gathers information for one training iteration.
        """
        self.entries.append( {
            'time' : time.clock(),
            'last_error' : err,
            'minimal_error' : training_results.minimal_error,
            'learning_rate' : method.n,
            'total_neurons' : nnc.total_neurons,
            } )

