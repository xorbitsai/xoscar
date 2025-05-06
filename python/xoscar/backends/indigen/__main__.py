if __name__ == "__main__":
    import click

    @click.group(
        invoke_without_command=True,
        name="xoscar",
        help="Xoscar command-line interface.",
    )
    def main():
        pass

    @main.command("start_sub_pool", help="Start a sub pool.")
    @click.option("shm_name", "-sn", type=str, help="Shared memory name.")
    def start_sub_pool(shm_name):
        from xoscar.backends.indigen.pool import MainActorPool

        MainActorPool._start_sub_pool_in_child(shm_name)

    main()
