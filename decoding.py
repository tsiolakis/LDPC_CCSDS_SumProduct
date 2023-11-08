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
                V[i, j] = np.log((1 + prod) / (1 - prod))

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
        
        print("Iteration No. {} ".format(iter_cnt+1, decode_x))
        diff_positions = np.where(transmitted != decode_x)
        print("Errors at positions: ", diff_positions)
        print("Number of errors: ", len(diff_positions[0]))
        print("----------------------")
        
        # Check Convergence Criterium
        if (H.dot(decode_x.transpose()) % 2 == 0).all():
            return decode_x, iter_cnt+1

        for j in range(num_check_nodes):
            for i in range(num_var_nodes):
                if H[i][j] == 1:
                    M[i][j] += L[j]
        
                    
    return decode_x, max_iter



matrix = np.array([
    [1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1]
])
transmitted = np.array([1, 1, 0, 0, 1, 1])
received = np.array([1, 1, 0, 1, 1, 1])
result, iter_cnt = ldpc_sum_product_decoder(matrix, received, transmitted, max_iter=20)

print("************Encoder Results************")
print("Transmitted = ", transmitted)
print("Received = ", received)
print("Decoder result = ", result)
print("Number of iterations required = ", iter_cnt)

