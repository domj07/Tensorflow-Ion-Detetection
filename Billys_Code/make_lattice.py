import numpy as np
from scipy.special import  ellipe, ellipeinc


"""
Script for making lattice for artificial data. 
Essentially it starts by building hexagonal shells, and then progressively makes the shells more and more elliptical as the shell number increases.
"""



def get_ellipse_circumference(a, b):
    m = 1-(b**2/a**2)
    return 4*a*ellipe(m)


# My own algorithm to analytically solve for the kth equidistant point on ellipse
# Perhaps there's a more efficient scipy alternative
def get_kth_point_on_ellipse(k, n, a, b, accuracy = 1e-4):
    if a == 0:
        return (0, 0)
    m = 1-(b**2/a**2)
    c = get_ellipse_circumference(a, b)
    expected = c*(k/n)

    # initial guess - as if its a circle
    phi = 2*np.pi*k/n

    error = (a*ellipeinc(phi, m) - expected)/c

    if error < 0:
        phi_under = phi
        phi_over = 2*np.pi
    elif error > 0:
        phi_under = 0
        phi_over = phi
    else:
        return np.array([a*np.sin(phi), b*np.cos(phi)])

    while abs(error) > accuracy:
        phi = (phi_over + phi_under)/2
        error = (a*ellipeinc(phi, m) - expected)/c

        if error < 0:
            phi_under = phi
        elif error > 0:
            phi_over = phi
        else:
            return np.array([a*np.sin(phi), b*np.cos(phi)])
    
    return np.array([a*np.sin(phi), b*np.cos(phi)])



# Elon is the elongation of the lattice. elon = 1 will build up shells around a line of 3 ions, instead of 1. 
# elon = 2 will build it around a line of 5 ions etc. Results in an elongated, elliptical lattice.
def make_lattice(shells, elon=0):
    theta_u = 0
    theta_v=60
    u = np.array([np.cos(theta_u/180*np.pi), np.sin(theta_u/180*np.pi)])
    v = np.array([np.cos(theta_v/180*np.pi), np.sin(theta_v/180*np.pi)])
    
    # Get lattice coordinates (shell-wise)
    lattice_coords = {key:[] for key in range(0, shells+1)}

    for shell in range(0, shells+1):
        # Corners
        i = 0
        shell_points = []
        while i <= shell:
            shell_points.append((-shell-elon, i+elon))
            shell_points.append((shell+elon, -i-elon))
            shell_points.append((-i-elon, shell+elon))
            shell_points.append((i+elon, -shell-elon))
            i+=1
        # Diagonals
        i = 0
        while i <= shell+elon:
            shell_points.append((-elon+i , (elon+shell)-i))
            shell_points.append((-(elon+shell)+i , elon-i))
            shell_points.append((elon-i , -(elon+shell)+i))
            shell_points.append(((elon+shell)-i , -elon+i))
            i += 1
        # Remove Duplicates
        shell_points = list(set(shell_points))
        # Save to dictionary
        lattice_coords[shell] = shell_points 


    # Translate these lattice coords into the hexagonal lattice
    positions = {key:[] for key in range(0, shells+1)}

    for shell, coords in lattice_coords.items():
        for i, j in coords:
            position = i*u + j*v
            positions[shell].append((position[0], position[1]))

    # Create rotation matrix to rotate the system so the elongation axis
    # is the x axis (easier to draw ellipses on this)
    theta = np.radians(60)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    rot_pos = {key:None for key in range(0, shells+1)}

    for shell, coords in positions.items():
        coords = np.array(coords)
        new_pos = np.matmul(R, coords.T).T
        rot_pos[shell] = new_pos

    
    # Create rounded-out lattice
    rounded_lattice = {key:None for key in range(0, shells+1)}

    for shell in range(0, shells+1):
        # Define best-fit ellipse
        a = (elon + shell)
        b = shell

        hex_points = rot_pos[shell]
        hex_points = np.round(hex_points, 8)

        # Sort them in anti-clockwise order
        hex_points = sorted(hex_points, key=lambda x: np.arctan2(x[1], x[0]))

        n = len(hex_points)
        ell_points = []
        for k in range(n):
            ell = get_kth_point_on_ellipse(k, n, a, b)
            ell_points.append(ell)
        
        # Sort them in anti-clockwise order
        ell_points = sorted(ell_points, key=lambda x: np.arctan2(x[1], x[0]))
        ell_points = np.round(ell_points, 8)

        altered_shell = []

        # Weight - adjust to some function of shell to alter the rate of transition from hexagonal to elliptical
        weight = (shell/(shells))**1
        for k in range(n):
            ellipse_coords = np.array(ell_points[k])
            hex_coords = np.array(hex_points[k])

            new_point = (ellipse_coords*weight + hex_coords*(1-weight))
            altered_shell.append(new_point)
        
        rounded_lattice[shell] = altered_shell
            
    xs = []
    ys = []
    for vals in rounded_lattice.values():
        xpos, ypos = zip(*vals)
        xs.extend(xpos)
        ys.extend(ypos)

    return (xs, ys)