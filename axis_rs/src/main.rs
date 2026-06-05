use wgpu::Backends;

fn main() {
    println!("||||||  Axis GPU Diagnostics  ||||||");

    let instance = wgpu::Instance::default();
    let adapters = axis_rs::wgpu_core::block_on(instance.enumerate_adapters(Backends::all()));

    if adapters.is_empty() {
        println!("No compatible GPU adapters found.");
        return;
    }

    println!("Found {} adapter(s):\n", adapters.len());

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("[Adapter {}]", i);
        println!("  Name:      {}", info.name);
        println!("  Backend:   {:?}", info.backend);
        println!("  Device:    {:?}", info.device_type);
        println!("  Driver:    {}", info.driver);
        println!();
    }

    println!("Diagnostics complete.");
}
