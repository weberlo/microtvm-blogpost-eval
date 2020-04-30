layout src
target remote localhost:3336
add-symbol-file tmp06do9kvr.obj
add-symbol-file tmp_w1bvvnc.obj
set $pc = UTVMInit
break UTVMDone

define print_utvm_args
    set $i = 0
    while $i < utvm_num_tasks
        set $j = 0
        eval "print \"TASK %d ARGS\"", $i
        eval "set $num_task_args = utvm_tasks[$i].num_args"
        print "num_args: %d", $num_task_args
        while $j < $num_task_args
            eval "set $num_bits = ((DLTensor*) utvm_tasks[0].arg_values[0].v_handle)->dtype.bits"
            if $num_bits == 8
                print "dtype: int8"
                eval "p/d *((int8_t*) ((DLTensor*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            if $num_bits == 32
                print "dtype: int32"
                eval "p/d *((int32_t*) ((DLTensor*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            set $j = $j + 1
        end
        set $i = $i + 1
    end
end

print_utvm_args
