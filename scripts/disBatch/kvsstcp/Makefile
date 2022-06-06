
FC=gfortran
CFLAGS=-g
FFLAGS=-g

SUPOBJS =  kvsSupport.o

PROGS = kvsLoop kvsTestf kvsTestWIc kvsTestWIf
all: $(PROGS)

kvsSupport.o: kvsSupport.c kvsSupport.h
	$(CC) $(CFLAGS) -Wall -c $<

kvsLoop: kvsLoop.o $(SUPOBJS)
	$(CC) $(CFLAGS) -o $@ $< $(SUPOBJS)

kvsTestWIc: kvsTestWIc.o $(SUPOBJS)
	$(CC) $(CFLAGS) -o $@ $< $(SUPOBJS)

kvsTestf: kvsTestf.o $(SUPOBJS)
	$(FC) $(FFLAGS) -o $@ $< $(SUPOBJS)

kvsTestWIf: kvsTestWIf.o $(SUPOBJS)
	$(FC) $(FFLAGS) -o $@ $< $(SUPOBJS)

clean:
	/bin/rm -f *.o $(PROGS)
