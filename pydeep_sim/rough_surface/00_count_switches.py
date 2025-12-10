def count_switches(matrix):
    """
    Count the number of switches between adjacent cells with different contact conditions.

    Parameters:
    matrix: 2D array of surface tractions

    Returns:
    int: Number of transitions between contact and non-contact
    """
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    switches = 0

    # Count horizontal switches (comparing each cell with its right neighbor)
    for i in range(rows):
        for j in range(cols - 1):
            if bool(abs(matrix[i][j]) < 1.0E-12) ^ bool(abs(matrix[i][j + 1]) < 1.0E-12):
                switches += 1

    # Count vertical switches (comparing each cell with its bottom neighbor)
    for i in range(rows - 1):
        for j in range(cols):
            if bool(abs(matrix[i][j]) < 1.0E-12) ^ bool(abs(matrix[i + 1][j]) < 1.0E-12):
                switches += 1

    return switches
