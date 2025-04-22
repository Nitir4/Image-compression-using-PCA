# PCA Implementation from scratch
def PCA(matrix, k):
    # Compute the mean of the matrix along each feature
    mean = np.mean(matrix, axis=0)
    
    # Center the matrix by subtracting the mean
    centered = matrix - mean
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order and select the top k eigenvectors
    sorted_idx = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_idx[:k]]
    
    # Project the original data onto the top k eigenvectors
    projected = np.dot(centered, top_eigenvectors)
    
    # Reconstruct the data from the projection
    reconstructed = np.dot(projected, top_eigenvectors.T) + mean
    
    return reconstructed, eigenvalues, sorted_idx
