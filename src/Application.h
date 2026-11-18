#pragma once

#include "BackendOptions.h"
#include "RenderParams.h"

#include <cstdint>
#include <string>

struct ApplicationOptions : BackendOptions {
	int width = 800;
	int height = 600;
	RenderParams params;
	std::string obj_path;
	std::string texture_path;
};

enum class OptionResult : std::uint8_t { Parsed, Invalid, Unknown };

bool parse_int_arg(const char* flag, const char* str, int& out);
void print_usage(const char* program_name);
void print_backend_usage();
OptionResult parse_backend_option(const std::string& arg, const char* value, ApplicationOptions& options);
int run_application(const ApplicationOptions& options, const char* program_name);
