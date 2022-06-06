      program kvsTest
      implicit none

      logical   	kvscheckown
C     cinfo is an opaque type holding KVS connection info.
      integer*4 	cinfo(3), rank, nclones, chunk /13/,
     $     limit /1234/, psum(3)
      integer   	i, sum /0/, work /0/
      integer		t0, time
      character*10	cchunk, climit

C     Disable kvs tracing.
      cinfo(3) = 0
C     Connect to the key/value service.
      call kvsconnect(cinfo)

C     Figure out our role in this run.
      call slurmme(rank, nclones)
      print '(A4,I4,A4,I6)',
     $     'Rank ', rank, ' of ', nclones
      
C     The work in this loop will be split among all participants into
C     blocks of size "chunk".
      t0 = time()
      do i = 0, limit-1
         if ( kvscheckown(cinfo, rank, nclones, chunk, 'ticket') ) then
            sum = sum + 2*i + 1
            work = work + 1
C     Arrange for some clones to be slower than others.
            call sleep(mod(rank,2)+1)
         endif
      enddo
      psum(1) = rank
      psum(2) = work
      psum(3) = sum

C     Merge partial results.
      call kvsput(cinfo, 'psum', psum, 4*3)

      if ( rank .eq. 0 ) then
         sum = 0
         do i = 1, nclones
            call kvsget(cinfo, 'psum', psum)
            print *, psum(1), ' ', psum(2), ' => ', psum(3)
            sum = sum + psum(3)
         enddo
         print *, 'Sum of the first ', limit, ' odd integers: ', sum
         print *, 'Elapsed: ', time() - t0
      endif
      stop
      end
