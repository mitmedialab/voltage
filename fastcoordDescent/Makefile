LIB_DIR = lib

default: fastCoordDescent

fastCoordDescent: setup.py fastCoordDescent.pyx $(LIB_DIR)/libfastCoordDescent.a
	pip install -e . && rm -f celldemix.cpp  && rm -Rf build

$(LIB_DIR)/libfastCoordDescent.a:
	make -j 30 -C $(LIB_DIR) libfastCoordDescent.a

clean:
	make -C $(LIB_DIR) clean
