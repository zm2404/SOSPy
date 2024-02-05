This package is a Python version of MATLAB TOOLBOX -- SOSTOOLS. 

This package contains functions based on **sympy** package. However, in case of a large number of decision variables, **sympy** may take a long time to parse the data.

Demos are in the folder [SOSPy_demos](https://github.com/zm2404/SOSPy/tree/main/SOSPy_demos/Sympy_demos).

### Updates in version 0.2.9:
- Add some data type annotations

### Updates in version 0.2.8:
- Fix bugs in **sossolve()**

### Updates in version 0.2.7:
- Fix bugs in **sosscssolver()**
- Add a function **remove_redundant_row()** to remove redundant rows in a non-full rank matrix, and get a full rank matrix
- Apply **remove_redundant_row()** to soscvxoptsolver() and sosscssolver(); now, they can solve demo 9
- Fix bugs in **sosgetsol()**; they can appropriately show the result

### Updates in version 0.2.6:
- Fix bugs in **soscvxoptsolver()** and **sosscssolver()**

### Updates in version 0.2.5:
- Allow changes of CVXOPT and SCS solver parameters through **options**

### Updates in version 0.2.4:
- Fix bugs in **sossolve()**

### Updates in version 0.2.3:
- Make SCS and CVXOPT independent from CVXPY. Now running SCS and CVXOPT doesn't need to go through CVXPY
- Adjust **sossolve()** accordingly
- Adjust **findsos()** accordingly


### Updates in version 0.2.2:
- Fix bugs in **sosmoseksolver()**


For technique issues, send to sospypython@gmail.com

Contributors: 
- James Anderson, email: james.anderson@columbia.edu
- Leonardo Felipe Toso, email: lt2879@columbia.edu
- Zhe Mo, email: zm2404@columbia.edu

