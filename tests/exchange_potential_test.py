import numpy as np
#test exchange potential 

def test_exchange_hydrogen(hydrogen_solver):
    ''' 
       the test aims to evaluate the realness of exchange potential 
       exchange potential -> is always negative and scale as (n:int)*(1/3)
       Test for hydrogen 
    '''   
    Vx = hydrogen_solver.exchange(1, 0) 
    assert np.all(Vx <= 0), "Exchange potential should be negative for hydrogen density"


def test_exchange_scaling(hydrogen_solver):
    """Exchange potential must scale as n^(1/3)"""
    Vx = hydrogen_solver.exchange(1, 0)

    # double density 
    hydrogen_solver.den *= 2.0
    Vx_scaled = hydrogen_solver.exchange(1, 0)

    scaling_factor = 2**(1/3)
    ratio = Vx_scaled / Vx

    mask = np.isfinite(ratio) & (np.abs(Vx) > 0)
    assert np.any(mask), "No valid grid points for scaling check"
    assert np.allclose(ratio[mask], scaling_factor, rtol=1e-2), \
        f"Scaling test failed: expected {scaling_factor}, got mean {np.mean(ratio[mask])}"