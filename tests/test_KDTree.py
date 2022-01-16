import pytest
import numpy as np
 

# Test Factored KD Trees.
# Test Run for numpy arrays. 
# Test Run for a list of vectors. 

def gen_random_transition(buffer):
    return [np.random.rand(*buffer.state_shape), 
           np.random.rand(*buffer.action_shape), 
           np.random.rand(*buffer.state_shape), 
           np.random.rand(1)[0],
           bool(np.random.randint(2))]


@pytest.fixture
def fixed_sa_pairs():
    """
    get a default set of sa pairs.
    """
    from lmdp.data.buffer import StandardBuffer
    import numpy as np 

    sa_pairs = [(np.array([0.2, 0.2, 0.2, 0.2]), [0]),
                (np.array([0.3, 0.3, 0.3, 0.3]), [0]),
                (np.array([0.4, 0.4, 0.4, 0.4]), [0]),
                (np.array([0.5, 0.5, 0.5, 0.5]), [0]),
                (np.array([0.6, 0.6, 0.6, 0.6]), [0]),
                (np.array([0.11, 0.11, 0.11, 0.11]), [1]),
                (np.array([0.12, 0.12, 0.12, 0.12]), [1]),
                (np.array([0.13, 0.13, 0.13, 0.13]), [1]),
                (np.array([0.14, 0.14, 0.14, 0.14]), [1]),
                (np.array([0.15, 0.15, 0.15, 0.15]), [1]),
                (np.array([0.116, 0.16, 0.16, 0.16]), [1]), ]
    return sa_pairs


@pytest.mark.buffer
@pytest.mark.standard_buffer
def test_standard_buffer(standard_buffer_instance):
    buffer = standard_buffer_instance
        
    # fill the buffer
    for i in range(buffer.max_size):
        buffer.add(*gen_random_transition(buffer))
        assert i + 1 == len(buffer)
    
    # fill the buffer some more
    for i in range(int(buffer.max_size/2)):        
        buffer.add(*gen_random_transition(buffer))
        # make sure FIFO replacement of the states. 
        assert buffer.ptr == i + 1
        assert len(buffer) == buffer.max_size
        
    assert buffer.state_shape == [2,2]