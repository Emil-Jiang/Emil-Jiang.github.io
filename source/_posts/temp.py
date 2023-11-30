from sympy import symbols, z, inverse_z_transform
# 定义变量和变换表达式
z = symbols('z')
X_z = 1/(z-1)

# 进行Z变换
x_n = inverse_z_transform(X_z, z, n)
x_n_frac = inverse_z_transform(1/X_z, z, n)

print(x_n,x_n_frac)