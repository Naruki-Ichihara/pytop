FROM ichiharanaruki/pytop:latest
ARG INSTALL_CLAUDE=true
RUN apt update
RUN apt upgrade -y
RUN apt -y install libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
RUN pip install --upgrade pip
RUN pip install pygmsh
RUN pip install meshio

# Downgrade numpy to 1.26.4 for petsc4py/pyadjoint compatibility
RUN pip install numpy==1.26.4

# Install dolfin-adjoint and create symlink for ufl_legacy
#RUN pip install dolfin-adjoint

RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://claude.ai/install.sh | bash; \
    fi


WORKDIR /home/
CMD ["/bin/bash"]