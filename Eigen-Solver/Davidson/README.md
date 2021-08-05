Extrapolating Davidson Method
==============================

About
-------------

This is a Python implementation on extrapolating block restarted Davidson method.


Requirements
---------------

1. numpy == 1.19.2
2. scipy == 1.5.1
3. matplotlib == 3.3.0
4. h5py == 2.10.0

Setup Environment
------------------

1. Run `pip install -r requirements.txt` to install dependencies


Usage
-------

To set up the algorithm for testing, a list of parameters need to be set properly before running:
1. --tol:                 tolerance of convergence for each eigen value-vector
2. --data_file_name:      data file name (directory)
3. --n_eig:               number of roots/eigen values to solve
4. --n_guess:             number of initial guess vectors
5. --k:                   number of steps Davidson method (i.e. k-step Davidson method)
6. --max_iter:            max number of iteration to restart Davidson method
7. --descent_order:       True or False, solve the max roots/eigen values if descent
8. --init:                initialization for guess vectors (currently only "random" or "Euclidean")
9. --gamma:               extrapolation parameter
10. --compare:            0: only original Davidson 1: only extrapolated Davidson 2: both original and extrapolation
11. --plot:               True or False, plot residual history

SYNOPSIS:
```
python test-Davidson-restarted.py [--tol] [--data_file_name] [--n_eig] [--n_guess] [--k] 
                                  [--max_iter] [--descent_order] [--init] [--gamma] [--compare] [--plot]

example: python test-Davidson-restarted.py --tol 1e-10 --data_file_name data/HBAR_rhf.npz --n_eig 10 --n_guess 10 --k 50 --max_iter 40 --descent_order False --init random --gamma -0.5 --compare 2 --plot True
```

Notice
----------------
1. File `setting.txt` list parameters for solving eigenvalue/vector pairs for different sparse matrices that I have tested on.
2. This is a research project in mathematics, any advice or suggestions from computer science literature are welcome to make this program better. There are sill lots of work left to be done, if you find any errors, feel free to let us know. 


Development
-----------

If you want to work on this application we’d love your pull requests on GitHub!

1. If you open up a ticket, please make sure it describes the problem or feature request fully.
2. If you send us a pull request, make sure you add a test for what you added, and make sure the full test suite runs with `make test`.


Contact
----------------
Name: Chi (Chris) Ding

Email: ding.chi@ufl.edu