typedef struct simulation_type {
    double *doublevar;
    int doublevar_shape[2];
    char *stringvar;
    int stringvar_len;
}c_simulation_type;
extern struct simulation_type* bind_simulation_init(int ny, int nx);
extern void bind_simulation_final(struct simulation_type *obj);
extern void bind_c2f_string(char* c_string, int c_string_len, char* f_string, int f_string_len);
extern void bind_f2c_string(char* f_string, int f_string_len, char* c_string, int c_string_len);
