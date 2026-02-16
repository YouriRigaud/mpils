require_relative "param_reader.rb"
#=== This wrapper reads in some parameters, sets up a call to CPLEX, performs it, reads the result, and outputs it in a nice format for ParamILS.

#=== Deal with inputs.
if ARGV.length < 5
	puts "CPLEX wrapper is a wrapper around CPLEX, so it can be tuned with ParamILS."
	puts "Usage: ruby cplex_wrapper.rb <instance_relname> <cutoff_time> <cutoff_length> <seed> <run_objective> <params to be passed on>."
	exit -1
end
instance_relname = ARGV[0]
instance_specifics = ARGV[1] # ignored in all cases so far
cutoff_time = ARGV[2].to_f
cutoff_length = ARGV[3].to_i 
seed = ARGV[4].to_i # ignored

if cutoff_length > 2100000000
	cutoff_length = 2100000000
end

log_inst_name = instance_relname[20,100]
#String1 = instance_relname
#String1.scan(/<([^>]*)>/).last.first
t = Time.now
datetime = t.strftime("%Y-%m-%d %H:%M:%S") # YYYY-MM-DD HH:MM:SS
datetime2 = t.strftime("%Y-%m-%d_%H:%M:%S") # YYYY-MM-DD_HH:MM:SS added by Youri
#outfile = "cplex-out-#{datetime}-#{rand}".gsub(/ /,"")
#outfile = "./example_cplex/outfiles/cplex-out-tmp-#{rand}"
outfile = "tuner_working_dir/solver/outfiles/cplex-out-tmp-#{datetime2}"
#logfile = "/Users/ilyashimmich/Documents/POSTDOC/projets_DLP/article/CODE/paramils2.3.8-source/example_cplex/log_file/cplex.log"
#logfile = "./example_cplex/log_file/cplex_#{log_inst_name}.log"
logfile = "tuner_working_dir/solver/cplex.log"
#logfile = "/tmp/trace_files"

obj_lower_bound = false
mpi_run = false

param_lines = []
i=5
while i<ARGV.length-1
	param = ARGV[i].sub(/^-/,"")
	#set_cmd = param.gsub(/_/," ")
	set_cmd = param
	#=== This parameter is present only when we tune on the lower bound
	if param == "obj_lower_bound"
		obj_lower_bound = true
		i+=2
	elsif param == "process_mpi"
		mpi_run = true
		process_rank = ARGV[i+1]
		i+=2
	#=== Exception for parameter that has a tuple as value
	elsif param == "simplex_perturbation"
		param_lines << "set #{set_cmd} #{ARGV[i+1]} #{ARGV[i+2]}"
		i+=3
	else
		param_lines << "set #{set_cmd} #{ARGV[i+1]}"
		i+=2
	end
end


if mpi_run
	logfile = "/tmp/MLILS_youri/log_files/cplex_#{log_inst_name}_process_#{process_rank}.log"
end

#=== Change to however you call CPLEX locally.
#=== This is a File.popen construct, because I need to pipe in all the parameters after calling CPLEX.
#=== (you can also do this as a double File.popen construct: call ruby on the command line to call CPLEX and output something; that something can be read in with File.popen again - this is what I used to do in the commented part below. But now I'm just going via the logfile.)
cmd = "ruby -e 'File.popen(\"/home/yorig/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex\",\"w\"){|file| "
#cmd = "ruby -e 'File.popen(\"/home/ibm/cplex-studio/22.1.2/cplex/bin/x86-64_linux/cplex\",\"w\"){|file| " #Added by youri

cplex_lines = []
cplex_lines << "set logfile #{logfile}"
cplex_lines << "read #{instance_relname}"
#cplex_lines << "set clocktype 2"
#cplex_lines << "set mip limits nodes #{cutoff_length}"
cplex_lines << "set timelimit #{cutoff_time}"

#=== Set parameters.
cplex_lines += param_lines

cplex_lines << "display settings changed"
cplex_lines << "opt"
#cplex_lines << "write x.sol"
cplex_lines << "quit"

cplex_lines.map{|line| cmd += "file.puts \"#{line}\"; "}
cmd += "}'"

File.open("#{logfile}", "a") { |f| f.write "Start_Time #{Time.now} =========== \n" }
#File.open("#{logfile}", "a") { f.write "#{Time.now} =========== \n" }
#File.write(" Start_Time #{logfile}", "#{Time.now}", mode: "a")


#puts "Current time : #{datetime} > #{logfile}"

puts "Calling: #{cmd} > #{outfile}"
system("#{cmd} > #{outfile}")

=begin
inner_exit = $?
puts "inner exit: #{inner_exit}"
puts "Outfile: #{outfile}"
=end

=begin
File.open(outfile, "w"){|out|
#	puts "Calling cmd: #{cmd}"
	File.popen(cmd){|file|
		while line = file.gets
#			puts line
			out.puts line
		end
	}
}
=end



#########################################################################
#===  Reading output.
#########################################################################

#===  Setting up variables for run output.
solved = "CRASHED"
seed = -1
best_sol = -1
best_length = -1
measured_runlength = -1
measured_runtime = -1

gap = 1e10
lower_bound = -1e10

File.open(outfile){|file|
	while line = file.gets
#			puts "Read line: #{line}"
		
		#########################################################################
		#===  Parsing CPLEX run output
		#########################################################################
		if line =~/(#{float_regexp})%/
			gap = $1.to_f
		end
			
		if line =~ /MIP\s*-\s*Integer optimal solution:\s*Objective\s*=\s*(#{float_regexp})/
			best_sol = $1
			solved = 'SAT'
		end

		if line =~ /MIP\s*-\s*Integer optimal,\s*tolerance\s*\(#{float_regexp}\/#{float_regexp}\):\s*Objective\s*=\s*(#{float_regexp})/
			best_sol = $1
			solved = 'SAT'
		end
		
		
		if line =~ /Solution time\s*=\s*(#{float_regexp})\s*sec\.\s*Iterations\s*=\s*(\d+)\s*Nodes\s*=\s*(\d+)/
			measured_runtime = $1
			iterations = $2
			measured_runlength = $3
		end

		if line =~ /Solution time\s*=\s*(#{float_regexp}) sec\.\s*Iterations =\s*(\d+)/
#			solved = 'SAT'
			measured_runtime = $1
			iterations = $2				
		end

		if line =~ /Solution time =\s*(#{float_regexp}) sec\./
			#solved = 'SAT'
			measured_runtime = $1
			iterations = 0
		end
		
		if line =~ /Optimal:\s*Objective =\s*#{float_regexp}/
			solved = 'SAT'
		end

		if line =~ /Infeasible/
			solved = 'UNSAT'
		end
		
		if line =~ /MIP\s*-\s*Time limit exceeded, integer feasible:\s*Objective\s*=\s*(#{float_regexp})/
			best_sol = $1
			solved = 'TIMEOUT'
		end
		
		if line =~ /MIP - Time limit exceeded, no integer solution./
			solved = 'TIMEOUT'
		end
		
		if line =~ /CPLEX Error  1001: Out of memory./
			solved = 'TIMEOUT'
		end
		
		if line =~ /CPLEX Error  3019: Failure to solve MIP subproblem./
			solved = 'TIMEOUT'
		end

		if line =~ /CPLEX Error/
			solved = 'TIMEOUT'
		end
		
		if line =~ /Time limit exceeded/
			solved = 'TIMEOUT'
		end

		if line =~ /MIP\s*-\s*Node limit exceeded, integer feasible:\s*Objective\s*=\s*(#{float_regexp})/  # added by Youri
			best_sol = $1
			solved = 'NODE_LIMIT'
		end

		if line =~ /MIP - Node limit exceeded, no integer solution./ #added by Youri
			solved = 'NODE_LIMIT'
		end

		if line =~ /Current MIP best bound\s*=\s*(#{float_regexp})/
			lower_bound = $1.to_f
		end
		
#			if line =~ /Filesize limit exceeded/
#				solved = 'TIMEOUT'
#			end

		if line =~ /Solution time =\s*(#{float_regexp})\s*sec\.\s*Iterations\s*=\s*(\d+)\s*Nodes\s*=\s*\((\d+)\)\s*\((\d+)\)/
			measured_runtime = $1
			iterations = $2
			measured_runlength = $3
		end
		
		raise "Error: Failed to initialize CPLEX environment" if line =~ /Failed to initialize CPLEX environment./
	end


	if obj_lower_bound
		best_sol = lower_bound
	else
		best_sol = gap # This is really what we want to minimize. 
	end

#		raise "Error: solved neither TIMEOUT nor SAT - probably parsing problem. Here's the complete output: #{content}"

	if solved == "CRASHED"
		puts "\n\n==============================================\n\nWARNING: CPLEX crashed -> most likely file not found or no license\n\n=======================\n"
		#=== You may want to catch this exception and try a rerun once a license frees up - that's what I do in my own experiments.
		raise "No such file or directory: CPLEX crashed -> most likely file not found or no license\n\n======================="
	else
		puts "Result for ParamILS: #{solved}, #{measured_runtime}, #{measured_runlength}, #{best_sol}, #{seed}"
	end
}
#File.delete(outfile)
