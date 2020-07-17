LIB_DIR = lib

default: preprocess

preprocess: setup.py preprocess.pyx $(LIB_DIR)/libpreprocess.a
	pip install -e . && rm -f celldemix.cpp  && rm -Rf build

$(LIB_DIR)/libpreprocess.a:
	make -C $(LIB_DIR) libpreprocess.a

clean:
	make -C $(LIB_DIR) clean
