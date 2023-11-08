import numpy as np

def ldpc_sum_product_decoder(H, y, transmitted, max_iter=5, p_e=0.2):
    # Compute the number of variable nodes and check nodes
    num_var_nodes, num_check_nodes = H.shape

    # Initialize the message vectors
    M = np.zeros((num_var_nodes, num_check_nodes))
    V = np.zeros((num_var_nodes, num_check_nodes))

    # Compute the initial likelihoods based on the BSC channel model
    p0 = np.log((1 - p_e) / p_e)
    p1 = np.log(p_e / (1 - p_e))
    L = p1 * y + p0 * (1 - y)

    for j in range(num_check_nodes):
        for i in range(num_var_nodes):
            if H[i][j] == 1:
                M[i][j] += L[j]

    # Start the decoding loop
    decode_x = np.zeros_like(L)
    for iter_cnt in range(max_iter):
        # Compute variable-to-check messages
        for i in range(num_var_nodes):
            non_zero_elements_in_H_i = np.nonzero(H[i, :])[0]
            
            for j in np.nonzero(H[i, :])[0]:
                
                # Compute the product of the incoming messages from all other check nodes connected to i
                prod = 1
                for j_prime in np.nonzero(H[i, :])[0]:
                    if j_prime != j:
                        prod *= np.tanh(0.5 * M[i, j_prime])
                # Compute the variable-to-check message
                V[i, j] = np.log((1 + prod + 1e-10) / (1 - prod + 1e-10))

        for j in range(num_check_nodes):
            total = 0
            for i in range(num_var_nodes):
                total += V[i][j]
            L[j] += total

        for i in range(num_check_nodes):
            if L[i] < 0:
                decode_x[i] = 1
            else:
                decode_x[i] = 0
        
        print("iter {}: decode code: {} ".format(iter_cnt+1, decode_x))
        diff_positions = np.where(transmitted != decode_x)
        print("Errors at positions: ", diff_positions)
        print("Number of errors: ", len(diff_positions[0]))
        print("----------------------")
        if (H.dot(decode_x.transpose()) % 2 == 0).all():
            return decode_x, iter_cnt+1

        for j in range(num_check_nodes):
            for i in range(num_var_nodes):
                if H[i][j] == 1:
                    M[i][j] += L[j]
        
                    
    return decode_x, max_iter



def print_difference(a, b):
    diff_positions = np.where(a != b)
    print("Errors at positions: ", diff_positions)
    print("Number of errors: ", len(diff_positions[0]))

def flip_bits(array, num_flips):
    """
    Flips num_flips randomly chosen bits in a numpy array.

    Args:
        array (numpy.ndarray): The array to modify.
        num_flips (int): The number of bits to flip.

    Returns:
        numpy.ndarray: The modified array.
    """
    # Select num_flips random indices in the array
    indices = np.random.choice(array.size, num_flips, replace=False)
    # Flip the selected bits
    array[indices] = 1 - array[indices]
    return array


import scipy.io
mat_data = scipy.io.loadmat('data.mat')
G = np.array(mat_data['G'])
H = np.array(mat_data['H45'])
input = np.array(mat_data['input'])
input = input.reshape(-1)
transmitted = np.array(mat_data['output'])
transmitted = transmitted.reshape(-1)

received = transmitted.copy()
# Flip 2 bits
received[1] = 1 - received[1]
received[4] = 1 - received[4]
print_difference(received,transmitted)

print("******Entering function******")
result, iter_cnt = ldpc_sum_product_decoder(H, received, transmitted, max_iter=50, p_e = 0.01)
print("*----------------------------------------*")
print("Encoded input to the decoder = ", received)
# print("Decoded message = ", result)
print("Number of required iterations = ", iter_cnt)
    
print_difference(transmitted,result)
