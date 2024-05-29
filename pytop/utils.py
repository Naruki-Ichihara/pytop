# 　_*_ coding: utf-8 _*_
'''Utilites for pytop, including: bridging between fenics variables to numpy array, or vice versa.'''

from fenics import *
from fenics_adjoint import *
import numpy as np
from typing import Callable, Iterable, Optional
from dataclasses import dataclass
try:
    import meshio
except ImportError:
    raise ImportError("meshio is not installed. Please install it by running: pip install meshio[all]")
import os

@dataclass
class MPI_Communicator:
    '''MPI communicator for parallel computing.
    
    Attributes:
        comm_world (MPI_Comm): MPI communicator.
        rank (int): Rank of the process.
        size (int): Number of processes.
    '''
    comm_world = MPI.comm_world
    rank = MPI.comm_world.rank
    size = MPI.comm_world.size

def import_external_mesh(mesh_file: str, mpi_comm: Optional[MPI_Communicator]=None) -> Mesh:
    '''Import a mesh from a file. Time series XDMF files are not supported yet.

    Args: (str)
        mesh_file: path to the mesh file.
        mpi_comm (Optional): MPI communicator.

    Returns: (Mesh)
        fenics mesh.

    This function used the meshio library to read the mesh file and then convert it to a fenics mesh.
    Spported file formats are:

    > [Abaqus](http://abaqus.software.polimi.it/v6.14/index.html) (`.inp`),
    > [ANSYS](https://www.ansys.com/ja-jp) msh (`.msh`),
    > [AVS-UCD](https://lanl.github.io/LaGriT/pages/docs/read_avs.html) (`.avs`),
    > [CGNS](https://cgns.github.io/) (`.cgns`),
    > [DOLFIN XML](https://manpages.ubuntu.com/manpages/jammy/en/man1/dolfin-convert.1.html) (`.xml`),
    > [Exodus](https://nschloe.github.io/meshio/exodus.pdf) (`.e`, `.exo`),
    > [FLAC3D](https://www.itascacg.com/software/flac3d) (`.f3grid`),
    > [H5M](https://www.mcs.anl.gov/~fathom/moab-docs/h5mmain.html) (`.h5m`),
    > [Kratos/MDPA](https://github.com/KratosMultiphysics/Kratos/wiki/Input-data) (`.mdpa`),
    > [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html) (`.mesh`, `.meshb`),
    > [MED/Salome](https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html) (`.med`),
    > [Nastran](https://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00) (bulk data, `.bdf`, `.fem`, `.nas`),
    > [Netgen](https://github.com/ngsolve/netgen) (`.vol`, `.vol.gz`),
    > [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#mesh-representation-of-segmented-object-surfaces),
    > [Gmsh](https://gmsh.info/doc/texinfo/gmsh.html#File-formats) (format versions 2.2, 4.0, and 4.1, `.msh`),
    > [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file) (`.obj`),
    > [OFF](https://segeval.cs.princeton.edu/public/off_format.html) (`.off`),
    > [PERMAS](https://www.intes.de) (`.post`, `.post.gz`, `.dato`, `.dato.gz`),
    > [PLY](<https://en.wikipedia.org/wiki/PLY_(file_format)>) (`.ply`),
    > [STL](<https://en.wikipedia.org/wiki/STL_(file_format)>) (`.stl`),
    > [Tecplot .dat](http://paulbourke.net/dataformats/tp/),
    > [TetGen .node/.ele](https://wias-berlin.de/software/tetgen/fformats.html),
    > [SVG](https://www.w3.org/TR/SVG/) (2D output only) (`.svg`),
    > [SU2](https://su2code.github.io/docs_v7/Mesh-File/) (`.su2`),
    > [UGRID](https://www.simcenter.msstate.edu/software/documentation/ug_io/3d_grid_file_type_ugrid.html) (`.ugrid`),
    > [VTK](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf) (`.vtk`),
    > [VTU](https://vtk.org/Wiki/VTK_XML_Formats) (`.vtu`),
    > [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) ([TIN](https://en.wikipedia.org/wiki/Triangulated_irregular_network)) (`.wkt`),
    > [XDMF](https://xdmf.org/index.php/XDMF_Model_and_Format) (`.xdmf`, `.xmf`).

    '''
    if mesh_file.endswith(".xdmf") or mesh_file.endswith(".xmf"):
        mesh = Mesh()
        with XDMFFile(mpi_comm, mesh_file) as infile:
            infile.read(mesh)
        return mesh

    mesh_original = meshio.read(mesh_file)
    mesh_original.write("temp.xml")
    mesh = Mesh("temp.xml")
    # Remove temporary file
    os.remove("temp.xml")
    return mesh

def make_noiman_boundary_domains(mesh: Mesh, subdomains: Iterable[SubDomain], initialize=False) -> Measure:
    '''Create a Measure object for the boundary domains.

    Args: (Mesh, Iterable[SubDomain], bool)
        mesh: fenics mesh.
        subdomains: list of subdomains.
        initialize: whether to initialize the boundary markers.

    Returns: (Measure)
        Measure object for the boundary domains.
    '''
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    if initialize:
        boundary_markers.set_all(0)
    for i, subdomain in enumerate(subdomains):
        subdomain.mark(boundary_markers, int(i+1))
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    return ds

def fenics_function_to_np_array(fenics_variable: Constant
                                               | Function
                                               | GenericVector ) -> np.ndarray:
    '''Convert fenics variable to numpy array.

    Args: (Constant | Function | GenericVector)
        fenics_variable: fenics values to be converted.

    Raises:
        TypeError: if the input is not a fenics vector.

    Returns: (np.ndarray)
        numpy array.
    '''
    if isinstance(fenics_variable, Constant):
        return np.array(fenics_variable.values())

    elif isinstance(fenics_variable, Function):
        fenics_vector = fenics_variable.vector()
        if fenics_vector.mpi_comm().size > 1:
            gathered_fenics_vector = fenics_vector.gather(
                np.arange(fenics_vector.size(), dtype='I'))
        else:
            gathered_fenics_vector = fenics_vector.get_local()
        return np.asarray(gathered_fenics_vector)

    elif isinstance(fenics_variable, GenericVector):
        if fenics_variable.mpi_comm().size > 1:
            gathered_fenics_vector = fenics_variable.gather(
                np.arange(fenics_variable.size(), dtype='I'))
        else:
            gathered_fenics_vector = fenics_variable.get_local()
        return np.asarray(gathered_fenics_vector)

    else:
        raise TypeError(
            'Input is not a supported type. Supported types are: Constant, Function, GenericVector on fenics.')


def np_array_to_fenics_function(np_array: np.ndarray,
                                fenics_function: Function) -> Function:
    '''Convert numpy array to fenics variable.

    Args: (np.ndarray, Function)

        np_array: numpy array to be converted.
        fenics_function: fenics function to be assigned.

    Raises:
        TypeError: if the input is not a numpy array.
        ValueError: if the input numpy array is not of the same size as the fenics vector.

    Returns: (Function)
        fenics variable.

    '''
    if isinstance(fenics_function, Function):
        function_space = fenics_function.function_space()
        function = type(fenics_function)(function_space)
        fenics_vector_size = function.vector().size()
        np_array_size = np_array.size
        if np_array_size != fenics_vector_size:
            err_msg = (
                f"Cannot convert numpy array to Function: Wrong size {np_array_size} vs {fenics_vector_size}")
            raise ValueError(err_msg)

        if np_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, but got {np_array.dtype}")
            raise ValueError(err_msg)

        range_begin, range_end = function.vector().local_range()
        localArray = np.asarray(np_array).reshape(
            fenics_vector_size)[range_begin:range_end]
        function.vector().set_local(localArray)
        function.vector().apply("insert")
        return function
    else:
        raise TypeError(
            'Input fenics vriable is not a supported type. Supported types is Function on fenics.')


def set_fields_to_fenics_function(fields: list[Callable[[Iterable], float]],
                                  function: Function) -> None:
    '''Set values for a fenics function.
    Elements of ```fields``` are assumed to be the followig pyfunction:
    ```python
    value1 = lambda x: f(x[0], x[1], ..., x[n]) # n is the dimension of the Function space.
    value2 = lambda x: g(x[0], x[1], ..., x[n]) 
    set_fields_to_fenics_function([value1, value2], function_space) # The rank of the function_space and length of the values must be the same.
    ```

    if the element is not a function but a constant value, it is assumed to be a constant value.

    ```python
    set_fields_to_fenics_function([1.0, 1.0], function_space)
    ```

    Args: (list, Function)
        values: list of values to be assigned.
        function: fenics function to be assigned.

    Raises:
        TypeError: if the input is not a list.

    '''
    if not isinstance(fields, list):
        raise TypeError('Input values must be a list.')

    class Field(UserExpression):
        def eval(self, value, x):
            for i, field in enumerate(fields):
                if not callable(field):
                    value[i] = field
                else:
                    value[i] = field(x)

        def value_shape(self):
            if len(fields) == 1:
                return ()
            else:
                return (len(fields),)

    function.interpolate(Field())
    return


def create_initialized_fenics_function(fields: list[Callable[[Iterable], float]],
                                       function_space: FunctionSpace) -> Function:
    '''Return a fenics function defined on the ```functionspace``` with values assigned.
    Elements of ```fields``` are assumed to be the following pyfunction:
    ```python
    value1 = lambda x: f(x[0], x[1], ..., x[n]) # n is the dimension of the Function space.
    value2 = lambda x: g(x[0], x[1], ..., x[n]) 
    create_initialized_fenics_function([value1, value2], functionspace) # The rank of the functionspace and length of the values must be the same.
    ```

    if the element is not a function but a constant value, it is assumed to be a constant value.

    ```python
    create_initialized_fenics_function([1.0, 1.0], functionspace)
    ```

    Args: (list, Function)
        values: list of values to be assigned.
        functionspace: fenics function space.

    Raises:
        TypeError: if the input is not a list.

    Returns: (Function)
        fenics function.

    '''
    if not isinstance(fields, list):
        raise TypeError('Input values must be a list.')
    function = Function(function_space)

    class Field(UserExpression):
        def eval(self, value, x):
            for i, field in enumerate(fields):
                if not callable(field):
                    value[i] = field
                else:
                    value[i] = field(x)

        def value_shape(self):
            if len(fields) == 1:
                return ()
            else:
                return (len(fields),)

    function.interpolate(Field())
    return function