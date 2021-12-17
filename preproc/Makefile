LIB_DIR = lib
LIB_NAME = libpreproc

all:
	make -C $(LIB_DIR)
	pip install -e . && rm -Rf build

clean:
	rm -fr $(LIB_NAME).cp* $(LIB_NAME).egg-info/
	make -C $(LIB_DIR) clean

