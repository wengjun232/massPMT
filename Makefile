export MAKE_TARGET=$@
export MAKE_SOURCE=$^
.PHONY:all
ith:=$(shell seq 0 300)

all: $(ith:%=result0/%)
result0/%:
	mkdir -p result0/$*
	#sed -n '$*,$*p' name >> result/$*/name
	python3 charge.py -i $*
	python3 tts.py -i $*

#all: $(ith:%=photon/save/%/general_singleMu.pdf) 
# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
