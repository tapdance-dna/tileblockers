def binary_division_iterator(start, end, depth_limit=None, start_depth=0):
    """
    Creates an iterator that yields two endpoint numbers, then their midpoint,
    then the midpoints between those, and so on.

    Parameters
    ----------
    start : float
        The starting endpoint
    end : float
        The ending endpoint
    depth_limit : int, optional
        Maximum depth of division. If None, continues indefinitely.
    start_depth : int, optional
        Depth at which to start yielding values. Default is 0.

    Yields
    ------
    float
        Numbers in the sequence of endpoints and midpoints

    Notes
    -----
    Written by Claude Sonnet 3.7 thinking, via Cursor.
    """
    # First yield the two endpoints if we're starting at depth 0
    if start_depth == 0:
        yield start
        yield end
    
    # If we're starting at a higher depth, we need to calculate all points up to that depth
    # but not yield them
    current_depth = 0
    intervals = [(start, end)]
    
    # Skip to the desired start depth
    while current_depth < (start_depth-1) and intervals:
        new_intervals = []
        for a, b in intervals:
            midpoint = (a + b) / 2
            new_intervals.append((a, midpoint))
            new_intervals.append((midpoint, b))
        intervals = new_intervals
        current_depth += 1
    
    # Now start yielding from the current depth
    while intervals and (depth_limit is None or current_depth < depth_limit):
        new_intervals = []
        for a, b in intervals:
            midpoint = (a + b) / 2
            yield midpoint
            new_intervals.append((a, midpoint))
            new_intervals.append((midpoint, b))
        intervals = new_intervals
        current_depth += 1

def binary_division_iterator_2d(start_x, end_x, start_y, end_y, depth_limit=None, start_depth=0):
    """
    Creates an iterator that explores a 2D parameter space through binary division,
    yielding corner points, then midpoints, then finer subdivisions.

    Parameters
    ----------
    start_x : float
        The starting x-coordinate
    end_x : float
        The ending x-coordinate
    start_y : float
        The starting y-coordinate
    end_y : float
        The ending y-coordinate
    depth_limit : int, optional
        Maximum depth of division. If None, continues indefinitely.
    start_depth : int, optional
        Depth at which to start yielding values. Default is 0.

    Yields
    ------
    tuple
        (x, y) coordinates in the sequence of points exploring the 2D space

    Notes
    -----
    The exploration pattern follows a quadtree-like subdivision of the 2D space,
    ensuring complete coverage of boundary points along both dimensions.
    """
    # Generate 1D point sequences for x and y axes
    x_points = set()
    y_points = set()
    
    # First yield the four corners if starting at depth 0
    if start_depth == 0:
        yield (start_x, start_y)
        yield (start_x, end_y)
        yield (end_x, start_y)
        yield (end_x, end_y)
        
        # Add these initial points to our sets
        x_points.update([start_x, end_x])
        y_points.update([start_y, end_y])
    
    # For higher start depths, calculate (but don't yield) the points we would have seen
    if start_depth > 0:
        # Simulate what points would be in x_points and y_points at the start_depth
        x_endpoints = [(start_x, end_x)]
        y_endpoints = [(start_y, end_y)]
        
        for d in range(start_depth):
            new_x_endpoints = []
            for a, b in x_endpoints:
                mid = (a + b) / 2
                x_points.add(a)
                x_points.add(b)
                if d < start_depth - 1:  # Don't add midpoints from the last step before start_depth
                    x_points.add(mid)
                new_x_endpoints.append((a, mid))
                new_x_endpoints.append((mid, b))
            x_endpoints = new_x_endpoints
            
            new_y_endpoints = []
            for a, b in y_endpoints:
                mid = (a + b) / 2
                y_points.add(a)
                y_points.add(b)
                if d < start_depth - 1:  # Don't add midpoints from the last step before start_depth
                    y_points.add(mid)
                new_y_endpoints.append((a, mid))
                new_y_endpoints.append((mid, b))
            y_endpoints = new_y_endpoints
    
    # Initialize current points and intervals for continued exploration
    x_intervals = [(start_x, end_x)]
    y_intervals = [(start_y, end_y)]
    current_depth = max(0, start_depth - 1)
    
    # Continue exploration from start_depth
    while (x_intervals or y_intervals) and (depth_limit is None or current_depth < depth_limit):
        # Process x intervals
        new_x_intervals = []
        for a, b in x_intervals:
            midpoint = (a + b) / 2
            if midpoint not in x_points:
                x_points.add(midpoint)
                # For each new x value, generate points with all known y values
                for y in y_points:
                    yield (midpoint, y)
                new_x_intervals.append((a, midpoint))
                new_x_intervals.append((midpoint, b))
        x_intervals = new_x_intervals
        
        # Process y intervals
        new_y_intervals = []
        for a, b in y_intervals:
            midpoint = (a + b) / 2
            if midpoint not in y_points:
                y_points.add(midpoint)
                # For each new y value, generate points with all known x values
                for x in x_points:
                    if (x, midpoint) not in [(x, y) for y in y_points if y != midpoint]:  # Avoid duplicates
                        yield (x, midpoint)
                new_y_intervals.append((a, midpoint))
                new_y_intervals.append((midpoint, b))
        y_intervals = new_y_intervals
        
        current_depth += 1
