layout src
target remote localhost:3333
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmphktdjxwb/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp5ogsu3_6/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmpq34chdg7/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmps39smmmn/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp2d_zyadr/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp0avh9ooc/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp6aasheh7/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmpn7ftz3fe/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp6ojyni80/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmpz0s856_x/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmppa_h6lq7/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmpfp18mac9/relocated.obj
add-symbol-file /var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tmp1owxxbr1/relocated.obj
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
            eval "set $num_bits = ((TVMArray*) utvm_tasks[0].arg_values[0].v_handle)->dtype.bits"
            if $num_bits == 8
                print "dtype: int8"
                eval "p/d *((int8_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            if $num_bits == 32
                print "dtype: int32"
                eval "p/d *((int32_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            set $j = $j + 1
        end
        set $i = $i + 1
    end
end

print_utvm_args
