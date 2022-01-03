      program kvsTest
      implicit none

      logical   	kvscheckown
C     cinfo is an opaque type holding KVS connection info.
      integer*4 	cinfo(3), rank, nclones, chunk, limit, psum(3)
      integer   	i, sum /0/, work /0/
      integer		t0, time
      character*10	cchunk, climit

C     Disable kvs tracing.
      cinfo(3) = 0
C     Connect to the key/value service.
      call kvsconnect(cinfo)

C     Figure out our role in this run.
      call slurmme(rank, nclones)
      print '(I4,A4,I4,A9,I6)',
     $     'Rank ', rank, ' of ', nclones
      
C     Get run parameters.
      if ( rank .eq. 0 ) then
C        Request input via KVS from a web inteface.
         call kvsget(cinfo, 'Loop limit?%10s', climit)
         read(climit, '(I10)') limit
         call kvsget(cinfo, 'Chunk?%10s', cchunk)
         read(cchunk, '(I10)') chunk
C        Send some HTML to the web inteface.
         call kvsput(cinfo, '@Run info',
     $        '<html><h2>Looping '//climit//
     $        ' times, in chunks of size '//cchunk//'.</h2>', 70)
C        Let the clones know.
         call kvsput(cinfo, 'limit', limit, 4)
         call kvsput(cinfo, 'chunk', chunk, 4)
      else
         call kvsview(cinfo, 'limit', limit)
         call kvsview(cinfo, 'chunk', chunk)
      endif

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
