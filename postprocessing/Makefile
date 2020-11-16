LIB_DIR = lib

default: postprocess

postprocess: setup.py postprocess.pyx $(LIB_DIR)/libpostprocess.a
	pip install -e . && rm -f postprocess.cpp  && rm -Rf build

$(LIB_DIR)/libpostprocess.a:
	make -j 30 -C $(LIB_DIR) libpostprocess.a

clean:
	make -C $(LIB_DIR) clean
