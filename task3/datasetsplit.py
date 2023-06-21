import sys, os
import time
import random; random.seed(0)
def main():
  if len(sys.argv)==1 or {"-h", "--help"} & set(sys.argv): log(main.__doc__,end="");exit(1)
  args=(a for a in sys.argv[1:] if not (a[:1]=="-" and a[1:]))
  kwargs={k: next(iter(v), True) for k, *v in(a.lstrip("-").split("=",1) for a in sys.argv[1:] if (a[:1]=="-" and a[1:]))}
  return run(*args, **kwargs)
def run(in_name, train_name, val_name, train_ratio=0.95):
  train_ratio= float(train_ratio)
  st= time.time()
  with open(in_name, encoding="utf-8-sig") if in_name!="-" else sys.stdin as in_s, \
    open(train_name, "w", encoding="utf-8-sig") if train_name!="-" else sys.stdout as train_s, \
    open(val_name, "w", encoding="utf-8-sig") if val_name!="-" else sys.stdout as val_s:
      example_lines= []
      total_train= total_val= 0
      i, next_line=-1, in_s.readline()
      while next_line!="":
        i += 1
        if(i)%1000 == 0: log_read(in_s, i)
        in_line, next_line= next_line, in_s.readline()
        example_lines.append(in_line)
        if in_line.strip() and next_line:continue
        random_fraction = random.random()
        if random_fraction<train_ratio:
          train_s.writelines(example_lines)
          total_train += 1
        else:
          val_s.writelines(example_lines)
          total_val += 1
        example_lines = []
  log(f"\x1b[1k\rprocessing: done. time: {time.time()-st:.3f}. total_train: {total_train}, total_val: {total_val}")
  log_read= lambda s, i="-", w="processing": \
    log("\x1b[1k\r%s:"%w, {"%.1f/%.1f MiB"%(s.tell()/2**20, os.path.getsize(s.fileno())/2**20) if s.seekable() else "%s it not allow"
if __name__== "__main__": main()       

          





