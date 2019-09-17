# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements class CpModel, the Changepoint model used by the 
Bayesian Online CP detection. The objects of this class store the prior 
information on the number of CPs as well as the prior distribution on the 
location of the CPs. 
Traditionally, both are contained in a single prior run-length 
distribution that implicitly specifies them, see Adams & MacKay (2007)
"""

from scipy import stats


class CpModel:
    """A model for location and number of the CPs in a (spatio-)temporal model.
    See Adams & MacKay (2007), or Fearnhead & Liu (2007) for details.
    The implemented model here is the geometric distribution
    
    Attributes:
        intensity: float >0; specifying the CP intensity rate

            
    Later additions:
        boundary: (optional) boolean; specifying if we want steady state 
            probability (True) or if we assume that there is a CP at 
            0 (False), with the latter being the default
        g: generic pmf (not necessarily geometric distribution) for CP
        g_0: generic pmf corresponding to boundary condition
        EPS: (static), with which precision we want to calculate the 
            resulting cdfs G and G_0 from g, g_0
    """
    
    def __init__(self, intensity):
        """Initialize the CP object by specifying intensity and indicating if
        you wish to use the steady state/advanced boundary conditions"""
        
        self.intensity = intensity
        self.cp_pmf = self.create_distribution()


    def create_distribution(self):
        """Creates the CP distribution using the object properties. 
        
        NOTE: At this point, it just operates with the geometric distribution.
        Once g and g_0 are allowed as input, this function handles the more
        general case, too."""
        
        return stats.geom(self.intensity)
    
    
    def pmf(self, k):
        """Returns the changepoint pmf for having k time periods passing 
        between two CPs.
        
        NOTE: At this point, it just operates with the geometric distribution.
        Once g and g_0 are allowed as input, this function handles the more
        general case, too."""
        
        return self.cp_pmf.pmf(k)
    
    def pmf_0(self, k):
        """Returns the cp pmf for the very first CP (i.e., we do not assume 
        that there is a CP at time point 0 with probability one)
        
        NOTE: At this point, it just operates with the geometric distribution.
        Once g and g_0 are allowed as input, this function handles the more
        general case, too."""

        if k ==0:
            return 1
        else:
            return 0
        
    
    def hazard(self, k):
        """Returns the changepoint hazard for r_t = k|r_{t-1} = k-1.
        
        NOTE: At this point, it just operates with the geometric distribution.
        Once g and g_0 are allowed as input, this function handles the more
        general case, too."""
        
        return 1.0/self.intensity     #memorylessness of geometric distribution
        
    def hazard_vector(self, k, m):
        """Returns the changepoint hazard for for r_t = l|r_{t-1} = l-1,  
        with l = k, k+1, k+2, ... m between two CPs. In the geometric case, 
        the hazard will be the same for all l.
        
        NOTE: At this point, it just operates with the geometric distribution.
        Once g and g_0 are allowed as input, this function handles the more
        general case, too."""
        return 1.0/self.intensity     #memorylessness of geometric distribution
    















