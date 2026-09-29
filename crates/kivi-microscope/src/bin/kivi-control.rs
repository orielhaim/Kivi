//! Runs one control server.
//!
//! ```text
//! cargo run --release -p kivi-microscope --bin kivi-control -- table-only 127.0.0.1:7001
//! ```
//!
//! The four controls are `noop`, `parse-only`, `hash-only` and `table-only`. Each
//! adds one stage to the one before it, so the difference between adjacent
//! servers is the marginal cost of that stage. Run the same benchmark against all
//! four and against the real server, and the cost of every layer of the read path
//! falls out of subtraction.
//!
//! These are instruments, not competitors. They implement no Redis semantics and
//! validate no replies; the question each answers is what a stage costs, not
//! whether the server is correct.

use std::net::SocketAddr;
use std::process::ExitCode;

use kivi_microscope::Control;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(name) = arguments.next() else {
        usage();
        return ExitCode::FAILURE;
    };
    let Some(address) = arguments.next() else {
        usage();
        return ExitCode::FAILURE;
    };
    let Some(control) = Control::from_name(&name) else {
        eprintln!("unknown control '{name}'");
        usage();
        return ExitCode::FAILURE;
    };
    let Ok(address) = address.parse::<SocketAddr>() else {
        eprintln!("'{address}' is not a socket address");
        return ExitCode::FAILURE;
    };

    if let Some(previous) = control.builds_on() {
        eprintln!(
            "reminder: this adds one stage to '{}'. Measure both and subtract; \
             a single control number says nothing on its own.",
            previous.name()
        );
    }
    if let Err(error) = kivi_microscope::control::serve(control, address) {
        eprintln!("control {} failed: {error}", control.name());
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn usage() {
    eprintln!("usage: kivi-control <noop|parse-only|hash-only|table-only> <addr:port>");
    eprintln!();
    eprintln!("the ladder, cheapest first:");
    for control in Control::LADDER {
        let added = control
            .builds_on()
            .map_or("baseline".to_owned(), |previous| {
                format!("+ {}", previous.name())
            });
        eprintln!("  {:<12} {added}", control.name());
    }
}
