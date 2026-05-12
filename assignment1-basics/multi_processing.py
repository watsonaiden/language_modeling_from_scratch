from multiprocessing import Process, Pipe

def f(conn):
    input = conn.recv()

    print(f'recieved "{input}"')
    conn.send([42, None, 'hello', input])
    while True:
        num = conn.recv()
        conn.send(num**2)


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()

    parent_conn.send('welcome user')
    print(parent_conn.recv())   # prints "[42, None, 'hello']"

    for i in [1,2,3,4]:
        parent_conn.send(i)
        result = parent_conn.recv()
        print(f'{i}^2 is {result}')

    p.terminate()
    p.join()
