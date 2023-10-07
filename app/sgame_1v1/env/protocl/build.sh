rm *.h *.cc *.o *.py
export LD_LIBRARY_PATH=/usr/local/lib/
protoc *.proto --python_out=.
#make clean;make -j8
